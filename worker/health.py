import os
from pathlib import Path

import grpc
import training_pb2 as pb
import training_pb2_grpc as rpc

data = Path(os.environ.get("DATA_DIR", "/shared/data"))
for name in ("objetos.csv", "recortes.npy", "recortes_labels.npy"):
    if not (data / name).is_file():
        raise SystemExit(f"Missing data: {name}")
with grpc.insecure_channel(os.environ.get("COORDINATOR", "coordinator:50051")) as channel:
    try:
        rpc.TrainingStub(channel).GetJobStatus(pb.JobRef(job_id="health"), timeout=2)
    except grpc.RpcError as error:
        if error.code() != grpc.StatusCode.NOT_FOUND:
            raise
