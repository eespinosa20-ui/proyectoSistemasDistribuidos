import json
import logging
import math
import os
import random
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc

import training_pb2 as pb
import training_pb2_grpc as rpc

ADDRESS = os.environ.get("COORDINATOR", "coordinator:50051")
DATA = Path(os.environ.get("DATA_DIR", "/shared/data"))
CHECKPOINTS = Path(os.environ.get("CHECKPOINT_DIR", "/shared/checkpoints"))
LIMITS = {"linear_regression": 4, "mlp": 2, "cnn": 1}
TRANSIENT = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}
LOG = logging.getLogger(__name__)


def retry(call, request, timeout=1):
    for attempt in range(4):
        try:
            return call(request, timeout=timeout)
        except grpc.RpcError as error:
            if error.code() not in TRANSIENT or attempt == 3:
                raise
            time.sleep(random.uniform(0, 0.2 * 2**attempt))


def execute():
    import torch
    from pipelines import cnn, lineal, mlp

    torch.set_num_threads(1)
    job = pb.Job.FromString(sys.stdin.buffer.read())
    config = {"job_id": job.job_id, "attempt": job.attempt, "seed": job.config.seed,
              "hyperparameters": {key: json.loads(value) for key, value in job.config.hyperparameters.items()}}
    assignment = {"worker_id": job.worker_id, "job_id": job.job_id, "attempt": job.attempt}
    pipelines = {"linear_regression": lineal, "mlp": mlp, "cnn": cnn}
    with grpc.insecure_channel(ADDRESS) as channel:
        stub = rpc.TrainingStub(channel)

        loss_sum = 0.0
        samples = 0

        def record_loss(module, inputs, output):
            nonlocal loss_sum, samples
            if isinstance(module, (torch.nn.MSELoss, torch.nn.CrossEntropyLoss)):
                batch_size = inputs[0].shape[0]
                loss_sum += output.detach().item() * batch_size
                samples += batch_size

        def progress(epoch, fraction):
            nonlocal loss_sum, samples
            average = loss_sum / samples if samples else None
            metrics = json.dumps({"loss": average}) if average is not None and math.isfinite(average) else ""
            retry(stub.ReportProgress, pb.Update(**assignment, epoch=epoch, progress=fraction,
                                                  metrics_json=metrics))
            loss_sum = 0.0
            samples = 0

        hook = torch.nn.modules.module.register_module_forward_hook(record_loss)
        try:
            result = pipelines[job.config.model_type].run(config, DATA, CHECKPOINTS, progress)
            update = pb.Update(**assignment, metrics_json=json.dumps(result["metrics"]),
                               model_path=result["model_path"])
        except grpc.RpcError:
            raise
        except Exception as error:
            LOG.exception("Job %s failed", job.job_id)
            update = pb.Update(**assignment, error=f"{type(error).__name__}: {error}")
        finally:
            hook.remove()
        retry(stub.CompleteJob, update)


def heartbeat(stub, worker, stop):
    while not stop.is_set():
        started = time.monotonic()
        try:
            try:
                retry(stub.Heartbeat, worker, timeout=0.4)
            except grpc.RpcError as error:
                if error.code() != grpc.StatusCode.NOT_FOUND:
                    raise
                retry(stub.RegisterWorker, worker, timeout=0.75)
        except grpc.RpcError as error:
            LOG.warning("Heartbeat: %s", error.code().name)
        stop.wait(max(0, 2 - (time.monotonic() - started)))


def launch(job):
    remaining = job.deadline_unix_ms / 1000 - time.time()
    if remaining <= 0:
        return
    try:
        subprocess.run([sys.executable, "-m", "main", "--execute"],
                       input=job.SerializeToString(), timeout=remaining, check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        LOG.warning("Job %s attempt %s ended: %s", job.job_id, job.attempt, error)


def consume(stub, worker, pool):
    slots = {model: threading.BoundedSemaphore(limit) for model, limit in LIMITS.items()}
    while True:
        available = [model for model, slot in slots.items() if slot.acquire(blocking=False)]
        if not available:
            time.sleep(0.1)
            continue
        assigned = None
        try:
            job = retry(stub.RequestJob, pb.Pull(worker_id=worker.worker_id,
                        request_id=uuid.uuid4().hex, model_types=available))
            if job.job_id:
                model = job.config.model_type
                future = pool.submit(launch, job)
                assigned = model
                future.add_done_callback(lambda _, slot=slots[model]: slot.release())
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.NOT_FOUND:
                LOG.warning("RequestJob: %s", error.code().name)
        finally:
            for model in available:
                if model != assigned:
                    slots[model].release()
        if assigned is None:
            time.sleep(0.25)


def main():
    while not all((DATA / name).is_file() for name in ("objetos.csv", "recortes.npy", "recortes_labels.npy")):
        time.sleep(1)
    worker = pb.Worker(worker_id=f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}")
    with grpc.insecure_channel(ADDRESS) as channel, ThreadPoolExecutor(sum(LIMITS.values())) as pool:
        stub = rpc.TrainingStub(channel)
        stop = threading.Event()
        pulse = threading.Thread(target=heartbeat, args=(stub, worker, stop), daemon=True)
        pulse.start()
        try:
            consume(stub, worker, pool)
        finally:
            stop.set()
            pulse.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if sys.argv[1:] == ["--execute"]:
        execute()
    else:
        main()
