import argparse
import json
import os
import tempfile
import time
from pathlib import Path

MODELS = {
    "linear_regression": {"variant": "ridge", "alpha": 0.1, "normalize": True},
    "mlp": {"hidden_layers": [16, 8], "epochs": 2},
    "cnn": {"num_filters": 4, "num_layers": 2, "epochs": 2, "image_size": 32},
}


def local(data):
    import torch
    from pipelines import cnn, lineal, mlp

    torch.set_num_threads(1)
    assert torch.version.cuda is None
    with tempfile.TemporaryDirectory() as root:
        for module, (model, hp) in zip((lineal, mlp, cnn), MODELS.items()):
            config = {"job_id": model, "seed": 42, "hyperparameters": hp}
            events = []
            result = module.run(config, data, root, lambda *event: events.append(event))
            assert Path(result["model_path"]).is_file()
            assert Path(result["model_path"]).with_name("metrics.json").is_file()
            assert set(result["metrics"]) == {"val", "test"}
            if model != "linear_regression":
                checkpoint = Path(root) / f"{model}.pt"
                assert torch.load(checkpoint, weights_only=True)["epoch"] == 2
                checkpoint.unlink()

                def interrupt(epoch, fraction):
                    if fraction == 1.0:
                        raise InterruptedError("simulated worker stop")

                try:
                    module.run(config, data, root, interrupt)
                except InterruptedError:
                    pass
                else:
                    raise AssertionError("Interruption was not triggered")
                assert torch.load(checkpoint, weights_only=True)["epoch"] == 1
                resumed_events = []
                resumed = module.run(config, data, root, lambda *event: resumed_events.append(event))
                assert resumed_events[0] == (1, 0.5)
                assert resumed["metrics"] == result["metrics"]
            print(model, json.dumps(result["metrics"]), flush=True)


def distributed(address, seed):
    import grpc
    import training_pb2 as pb
    import training_pb2_grpc as rpc

    with grpc.insecure_channel(address) as channel:
        stub = rpc.TrainingStub(channel)
        try:
            stub.SubmitJob(pb.JobConfig(model_type="unsupported", data_version="mock-v1"), timeout=5)
        except grpc.RpcError as error:
            assert error.code() == grpc.StatusCode.INVALID_ARGUMENT
        else:
            raise AssertionError("Unsupported model accepted")
        submitted = {}
        for model, hp in MODELS.items():
            config = pb.JobConfig(model_type=model, data_version="mock-v1", seed=seed,
                                  hyperparameters={key: json.dumps(value) for key, value in hp.items()})
            job = stub.SubmitJob(config, timeout=5)
            assert stub.SubmitJob(config, timeout=5).job_id == job.job_id
            submitted[job.job_id] = model
        end = time.monotonic() + 90
        while submitted and time.monotonic() < end:
            for job_id, model in list(submitted.items()):
                job = stub.GetJobStatus(pb.JobRef(job_id=job_id), timeout=5)
                if job.state == pb.FAILED:
                    raise RuntimeError(f"{model}: {job.error}")
                if job.state == pb.COMPLETED:
                    assert Path(job.model_path).is_file()
                    assert Path(job.model_path).with_name("metrics.json").is_file()
                    assert set(json.loads(job.metrics_json)) == {"val", "test"}
                    print(model, pb.State.Name(job.state), job.worker_id, job.job_id, job.metrics_json, flush=True)
                    del submitted[job_id]
            if submitted:
                time.sleep(0.5)
        if submitted:
            raise TimeoutError(submitted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("local", "distributed"))
    parser.add_argument("--data", default=os.environ.get("DATA_DIR", "/shared/data"))
    parser.add_argument("--address", default=os.environ.get("COORDINATOR", "coordinator:50051"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "local":
        local(args.data)
    else:
        distributed(args.address, args.seed)
