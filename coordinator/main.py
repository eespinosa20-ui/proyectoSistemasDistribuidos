import hashlib
import itertools
import json
import logging
import os
import queue
import socket
import sys
import threading
import time
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path

import grpc
from google.protobuf.json_format import MessageToDict, ParseDict

import training_pb2 as pb
import training_pb2_grpc as rpc

DEADLINES = {"linear_regression": 30, "mlp": 180, "cnn": 600}
HEARTBEAT_LIMIT = 6


def invalid_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def identify(config):
    if config.model_type not in DEADLINES:
        raise ValueError("unsupported model_type")
    if not config.data_version:
        raise ValueError("data_version is required")
    params = {key: json.loads(value, parse_constant=invalid_constant)
              for key, value in config.hyperparameters.items()}
    encoded = json.dumps(dict(model_type=config.model_type, hyperparameters=params,
                              data_version=config.data_version, seed=config.seed),
                         sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass
class Worker:
    seen: float
    alive: bool = True
    request_id: str = ""
    job_id: str = ""
    attempt: int = 0


class Coordinator(rpc.TrainingServicer):
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.workers = {}
        self.listeners = {}
        self.stream_slots = threading.BoundedSemaphore(4)
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            saved = {"jobs": [], "reassignments": 0}
        legacy = isinstance(saved, list)
        records = saved if legacy else saved["jobs"]
        jobs = [ParseDict(record, pb.Job()) for record in records]
        self.jobs = {job.job_id: job for job in jobs}
        self.identities = {identify(job.config): job.job_id for job in jobs}
        self.reassignments = (
            sum(max(0, job.attempt - 1) + int(job.state == pb.REASSIGNED) for job in jobs)
            if legacy else saved["reassignments"]
        )
        recovered = [self.requeue(job) for job in jobs if job.state in (pb.RUNNING, pb.REASSIGNED)]
        if recovered:
            self.commit(recovered, sum(job.state == pb.RUNNING for job in jobs))

    @staticmethod
    def requeue(job):
        clone = pb.Job()
        clone.CopyFrom(job)
        clone.state = pb.PENDING
        clone.worker_id = ""
        clone.deadline_unix_ms = 0
        clone.epoch = 0
        clone.progress = 0
        clone.metrics_json = ""
        return clone

    def emit(self, event):
        for listener, overflow in list(self.listeners.items()):
            try:
                listener.put_nowait(event)
            except queue.Full:
                overflow.set()
                del self.listeners[listener]

    def commit(self, changes, reassigned=0, context=None):
        jobs = self.jobs.copy()
        jobs.update((job.job_id, job) for job in changes)
        count = self.reassignments + reassigned
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as file:
                json.dump({"jobs": [MessageToDict(job, preserving_proto_field_name=True)
                                     for job in jobs.values()], "reassignments": count}, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        except OSError:
            if context is not None:
                logging.exception("Could not persist queue")
                context.abort(grpc.StatusCode.INTERNAL, "could not persist queue")
            raise
        self.jobs = jobs
        self.reassignments = count
        for job in changes:
            self.emit(pb.Event(kind="job", job=job, worker_id=job.worker_id))
        if reassigned:
            self.emit(pb.Event(kind=f"reassignments:{count}"))

    def submit(self, configs, context):
        try:
            identified = [(identify(config), config) for config in configs]
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        with self.lock:
            changes = {}
            result = []
            for identity, config in identified:
                existing = self.jobs.get(self.identities.get(identity))
                if existing is None:
                    existing = changes.setdefault(identity, pb.Job(job_id=identity, config=config))
                result.append(existing)
            if changes:
                self.commit(list(changes.values()), context=context)
                self.identities.update((identity, job.job_id) for identity, job in changes.items())
            return result

    def SubmitJob(self, request, context):
        return self.submit([request], context)[0]

    def SubmitGridSearch(self, request, context):
        try:
            identify(request.base)
        except ValueError as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        names = sorted(request.axes)
        size = 1
        for name in names:
            size *= len(request.axes[name].json)
            if size == 0 or size > 10_000:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "grid must contain 1 to 10000 combinations")
        configs = []
        for combination in itertools.product(*(request.axes[name].json for name in names)):
            config = pb.JobConfig()
            config.CopyFrom(request.base)
            config.hyperparameters.update(zip(names, combination))
            configs.append(config)
        return pb.Jobs(jobs=self.submit(configs, context))

    def GetJobStatus(self, request, context):
        with self.lock:
            job = self.jobs.get(request.job_id)
            if job is None:
                context.abort(grpc.StatusCode.NOT_FOUND, "job not found")
            return job

    def RegisterWorker(self, request, context):
        if not request.worker_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "worker_id is required")
        with self.lock:
            now = time.monotonic()
            self.expire(now, time.time(), context)
            worker = self.workers.get(request.worker_id)
            if worker is None or not worker.alive:
                worker = self.workers[request.worker_id] = Worker(now)
                self.emit(pb.Event(kind="worker_registered", worker_id=request.worker_id))
            worker.seen = now
        return pb.Empty()

    def Heartbeat(self, request, context):
        with self.lock:
            now = time.monotonic()
            worker = self.workers.get(request.worker_id)
            if worker is None or not worker.alive or now - worker.seen > HEARTBEAT_LIMIT:
                context.abort(grpc.StatusCode.NOT_FOUND, "worker must register")
            worker.seen = now
        return pb.Empty()

    def RequestJob(self, request, context):
        if not request.request_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")
        with self.lock:
            now = time.time()
            self.expire(time.monotonic(), now, context)
            worker = self.workers.get(request.worker_id)
            if worker is None or not worker.alive:
                context.abort(grpc.StatusCode.NOT_FOUND, "worker must register")
            if worker.request_id == request.request_id:
                job = self.jobs.get(worker.job_id)
                if job and job.state == pb.RUNNING and job.attempt == worker.attempt:
                    return job
                return pb.Job()
            models = set(request.model_types)
            for job in self.jobs.values():
                if job.state == pb.PENDING and job.config.model_type in models:
                    assigned = pb.Job()
                    assigned.CopyFrom(job)
                    assigned.state = pb.RUNNING
                    assigned.worker_id = request.worker_id
                    assigned.attempt += 1
                    assigned.deadline_unix_ms = int((now + DEADLINES[job.config.model_type]) * 1000)
                    self.commit([assigned], context=context)
                    worker.request_id, worker.job_id, worker.attempt = (
                        request.request_id, assigned.job_id, assigned.attempt)
                    return assigned
            worker.request_id, worker.job_id, worker.attempt = request.request_id, "", 0
            return pb.Job()

    def update(self, request, context, complete):
        with self.lock:
            job = self.jobs.get(request.job_id)
            if job is None:
                context.abort(grpc.StatusCode.NOT_FOUND, "job not found")
            owns_job = (job.worker_id == request.worker_id and job.attempt == request.attempt)
            if owns_job and complete and job.state in (pb.COMPLETED, pb.FAILED):
                return pb.Empty()
            worker = self.workers.get(request.worker_id)
            if (not owns_job or job.state != pb.RUNNING or worker is None or not worker.alive
                    or time.monotonic() - worker.seen > HEARTBEAT_LIMIT
                    or time.time() * 1000 >= job.deadline_unix_ms):
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "assignment expired")
            updated = pb.Job()
            updated.CopyFrom(job)
            if complete:
                updated.deadline_unix_ms = 0
                updated.error = request.error
                updated.state = pb.FAILED if request.error else pb.COMPLETED
                if not request.error:
                    updated.progress = 1
                    updated.metrics_json = request.metrics_json
                    updated.model_path = request.model_path
            else:
                if not 0 <= request.progress <= 1:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "progress outside [0,1]")
                if request.progress >= job.progress:
                    updated.epoch, updated.progress = request.epoch, request.progress
                    if request.metrics_json:
                        updated.metrics_json = request.metrics_json
            if updated != job:
                self.commit([updated], context=context)
        return pb.Empty()

    def ReportProgress(self, request, context):
        return self.update(request, context, False)

    def CompleteJob(self, request, context):
        return self.update(request, context, True)

    def expire(self, monotonic_now, wall_now, context=None):
        with self.lock:
            dead = {name for name, worker in self.workers.items()
                    if worker.alive and monotonic_now - worker.seen > HEARTBEAT_LIMIT}
            changes = [self.requeue(job) for job in self.jobs.values()
                       if job.state == pb.RUNNING and
                       (job.worker_id in dead or wall_now * 1000 >= job.deadline_unix_ms)]
            if changes:
                self.commit(changes, len(changes), context)
            for name in dead:
                self.workers[name].alive = False
                self.emit(pb.Event(kind="worker_dead", worker_id=name))

    def watch(self, stop):
        while not stop.wait(1):
            try:
                self.expire(time.monotonic(), time.time())
            except OSError:
                logging.exception("Could not persist expired assignments")

    def StreamEvents(self, request, context):
        if not self.stream_slots.acquire(blocking=False):
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "too many event streams")
        listener = queue.Queue(maxsize=128)
        overflow = threading.Event()
        try:
            with self.lock:
                snapshot = [pb.Event(kind="snapshot")]
                snapshot.extend(pb.Event(kind="job", job=job, worker_id=job.worker_id)
                                for job in self.jobs.values())
                snapshot.extend(pb.Event(kind="worker_registered" if worker.alive else "worker_dead",
                                         worker_id=name) for name, worker in self.workers.items())
                snapshot.append(pb.Event(kind=f"reassignments:{self.reassignments}"))
                snapshot.append(pb.Event(kind="snapshot_end"))
                self.listeners[listener] = overflow
            yield from snapshot
            while context.is_active():
                if overflow.is_set():
                    context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "event consumer fell behind")
                try:
                    event = listener.get(timeout=0.5)
                except queue.Empty:
                    continue
                yield event
        finally:
            with self.lock:
                self.listeners.pop(listener, None)
            self.stream_slots.release()


def main():
    port = int(os.environ.get("PORT", "50051"))
    if sys.argv[1:] == ["health"]:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return 0
        except OSError:
            return 1
    service = Coordinator(os.environ.get("QUEUE_PATH", "/state/queue.json"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    rpc.add_TrainingServicer_to_server(service, server)
    server.add_insecure_port(f"[::]:{port}")
    stop = threading.Event()
    watcher = threading.Thread(target=service.watch, args=(stop,), daemon=True)
    server.start()
    watcher.start()
    logging.info("Coordinator listening on port %s", port)
    try:
        server.wait_for_termination()
    finally:
        stop.set()
        server.stop(0).wait()
        watcher.join()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

