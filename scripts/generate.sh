#!/bin/sh
set -eu
go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.35.2
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1
export PATH="$(go env GOPATH)/bin:$PATH"
python -m grpc_tools.protoc -I proto --go_out=proto --go_opt=paths=source_relative --go-grpc_out=proto --go-grpc_opt=paths=source_relative --python_out=worker --grpc_python_out=worker proto/training.proto
cp worker/training_pb2*.py dashboard/
go mod tidy
