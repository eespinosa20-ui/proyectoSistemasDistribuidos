#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python -m grpc_tools.protoc -I proto --python_out=worker --grpc_python_out=worker proto/training.proto
for component in coordinator client dashboard; do
  cp worker/training_pb2.py worker/training_pb2_grpc.py "$component/"
done
