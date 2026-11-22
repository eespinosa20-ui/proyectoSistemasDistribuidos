import argparse
import csv
import itertools
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import grpc

import training_pb2 as pb
import training_pb2_grpc as rpc

MODELS = ("linear_regression", "mlp", "cnn")
TRANSIENT = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}
REFERENCE = {
    "linear_regression": {"variant": ["ridge", "lasso"], "alpha": [0.01, 0.1, 1.0, 10.0]},
    "mlp": {"hidden_layers": [[16, 8], [32, 16], [64, 32, 16]],
            "learning_rate": [0.0001, 0.001, 0.01], "epochs": [2, 5]},
    "cnn": {"num_filters": [4, 8], "learning_rate": [0.001, 0.01], "image_size": [16, 32]},
}


def retry(call, request):
    for attempt in range(5):
        try:
            return call(request, timeout=5)
        except grpc.RpcError as error:
            if error.code() not in TRANSIENT or attempt == 4:
                raise
            time.sleep(random.uniform(0, 0.25 * 2**attempt))


def config(model, params, data_version, seed):
    return pb.JobConfig(model_type=model, data_version=data_version, seed=seed,
                        hyperparameters={key: json.dumps(value, allow_nan=False)
                                         for key, value in params.items()})


def job_record(job):
    return {"job_id": job.job_id, "model_type": job.config.model_type,
            "hyperparameters": {key: json.loads(value)
                                for key, value in job.config.hyperparameters.items()},
            "state": pb.State.Name(job.state), "epoch": job.epoch, "progress": job.progress,
            "worker_id": job.worker_id, "attempt": job.attempt,
            "metrics": json.loads(job.metrics_json) if job.metrics_json else None,
            "model_path": job.model_path, "error": job.error}


def submit(stub, args):
    base = config(args.model, args.params, args.data_version, args.seed)
    if args.grid is None:
        return job_record(retry(stub.SubmitJob, base))
    grid = pb.Grid(base=base)
    for key, values in args.grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid axis {key!r} must be a nonempty JSON array")
        grid.axes[key].json.extend(json.dumps(value, allow_nan=False) for value in values)
    return [job_record(job) for job in retry(stub.SubmitGridSearch, grid).jobs]


def reference_jobs():
    return [(model, dict(zip(axes, values)))
            for model, axes in REFERENCE.items()
            for values in itertools.product(*axes.values())]


def baseline(args, jobs):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
    import torch
    from pipelines import cnn, lineal, mlp

    torch.set_num_threads(1)
    if torch.version.cuda is not None:
        raise ValueError("Install the CPU-only worker requirements for the baseline")
    pipelines = dict(zip(MODELS, (lineal, mlp, cnn)))
    results = []
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        start = time.monotonic()
        for index, (model, params) in enumerate(jobs):
            job = {"job_id": f"baseline-{index + 1}", "seed": args.seed,
                   "hyperparameters": params}
            result = pipelines[model].run(job, args.data, checkpoint_dir, lambda *_: None)
            results.append({"job_id": job["job_id"], "model_type": model,
                            "hyperparameters": params, "state": "COMPLETED",
                            "metrics": result["metrics"]})
        elapsed = time.monotonic() - start
    return {"seconds": elapsed, "jobs": results}


def distributed(stub, args, jobs):
    start = time.monotonic()
    submitted = [retry(stub.SubmitJob, config(model, params, args.data_version, args.seed))
                 for model, params in jobs]
    reused = [job.job_id for job in submitted if job.attempt or job.state != pb.PENDING]
    pending = {job.job_id for job in submitted}
    finished = {}
    while pending:
        if time.monotonic() - start >= args.timeout:
            raise TimeoutError(f"Benchmark timed out with {len(pending)} unfinished jobs")
        for job_id in list(pending):
            job = retry(stub.GetJobStatus, pb.JobRef(job_id=job_id))
            if job.state in (pb.COMPLETED, pb.FAILED):
                finished[job_id] = job_record(job)
                pending.remove(job_id)
        if pending:
            time.sleep(0.25)
    return {"seconds": time.monotonic() - start, "reused_job_ids": reused,
            "jobs": [finished[job.job_id] for job in submitted]}


def benchmark(args):
    if not (args.baseline or args.distributed):
        raise ValueError("Choose --baseline, --distributed, or both")
    jobs = reference_jobs()
    json_path = args.output.with_suffix(".json")
    csv_path = args.output.with_suffix(".csv")
    settings = {"seed": args.seed, "data_version": args.data_version, "grid": REFERENCE}
    report = {"settings": settings}
    if json_path.exists() and not (args.baseline and args.distributed):
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        if previous["settings"] == settings:
            report = previous
    if args.baseline:
        report["baseline"] = baseline(args, jobs)
    if args.distributed:
        with grpc.insecure_channel(args.address) as channel:
            report["distributed"] = distributed(rpc.TrainingStub(channel), args, jobs)

    base = report.get("baseline")
    remote = report.get("distributed")
    failed = bool(remote and any(job["state"] == "FAILED" for job in remote["jobs"]))
    reduction = None
    if base and remote and not failed and not remote["reused_job_ids"]:
        reduction = (base["seconds"] - remote["seconds"]) / base["seconds"] * 100
    report["reduction_percent"] = reduction
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fields = ("mode", "seconds", "reduction_percent", "job_id", "model_type",
                  "state", "hyperparameters", "metrics", "error")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for mode in ("baseline", "distributed"):
            if mode in report:
                for job in report[mode]["jobs"]:
                    writer.writerow(dict(mode=mode, seconds=report[mode]["seconds"],
                                         reduction_percent=reduction, job_id=job["job_id"],
                                         model_type=job["model_type"], state=job["state"],
                                         hyperparameters=json.dumps(job["hyperparameters"]),
                                         metrics=json.dumps(job["metrics"]), error=job.get("error", "")))
    print(json.dumps({"json": str(json_path), "csv": str(csv_path),
                      "baseline_seconds": base["seconds"] if base else None,
                      "distributed_seconds": remote["seconds"] if remote else None,
                      "reused_jobs": len(remote["reused_job_ids"]) if remote else 0,
                      "reduction_percent": reduction}, indent=2))
    return int(failed)


def json_object(text):
    try:
        value = json.loads(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return value


def main():
    parser = argparse.ArgumentParser(description="Distributed training client")
    parser.add_argument("--address", default=os.getenv("COORDINATOR", "127.0.0.1:50051"))
    commands = parser.add_subparsers(dest="command", required=True)
    submission = commands.add_parser("submit")
    submission.add_argument("model", choices=MODELS)
    submission.add_argument("--params", type=json_object, default={})
    submission.add_argument("--grid", type=json_object)
    status = commands.add_parser("status")
    status.add_argument("job_ids", nargs="+")
    bench = commands.add_parser("benchmark")
    bench.add_argument("--baseline", action="store_true")
    bench.add_argument("--distributed", action="store_true")
    bench.add_argument("--data", type=Path, default=Path("data/processed"))
    bench.add_argument("--output", type=Path, default=Path("benchmark"))
    bench.add_argument("--timeout", type=float, default=3600)
    for command in (submission, bench):
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--data-version", default="mock-v1")
    args = parser.parse_args()
    if hasattr(args, "seed") and not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be an unsigned 32-bit integer")
    try:
        if args.command == "benchmark":
            return benchmark(args)
        with grpc.insecure_channel(args.address) as channel:
            stub = rpc.TrainingStub(channel)
            result = (submit(stub, args) if args.command == "submit" else
                      [job_record(retry(stub.GetJobStatus, pb.JobRef(job_id=job_id)))
                       for job_id in args.job_ids])
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except grpc.RpcError as error:
        print(f"{error.code().name}: {error.details()}", file=sys.stderr)
        return 1
    except (ValueError, OSError, ImportError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

