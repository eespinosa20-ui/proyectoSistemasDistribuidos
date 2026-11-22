import asyncio
import json
import os
import random
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import grpc
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

import training_pb2 as pb
import training_pb2_grpc as rpc


class Feed:
    def __init__(self):
        self.jobs = {}
        self.workers = {}
        self.reassignments = 0
        self.connected = False
        self.listeners = set()

    def notify(self):
        for listener in self.listeners:
            listener.set()

    def snapshot(self):
        counts = dict.fromkeys(("PENDING", "RUNNING", "COMPLETED", "FAILED"), 0)
        counts["REASSIGNED"] = self.reassignments
        assignments = {worker: [] for worker in self.workers}
        for job in self.jobs.values():
            counts[job["state"]] += 1
            if job["state"] == "RUNNING":
                assignments.setdefault(job["worker_id"], []).append(job["job_id"])
        return {"connected": self.connected, "counts": counts, "jobs": list(self.jobs.values()),
                "workers": [{"id": worker, "state": state, "jobs": assignments[worker]}
                            for worker, state in self.workers.items()]}

    async def consume(self):
        address = os.getenv("COORDINATOR", "coordinator:50051")
        delay = 0.5
        async with grpc.aio.insecure_channel(address) as channel:
            stub = rpc.TrainingStub(channel)
            while True:
                try:
                    jobs, workers, count = {}, {}, 0
                    async for event in stub.StreamEvents(pb.Empty()):
                        kind = event.kind
                        if kind == "job":
                            job = event.job
                            try:
                                metrics = json.loads(job.metrics_json) if job.metrics_json else None
                                json.dumps(metrics, allow_nan=False)
                            except ValueError:
                                metrics = job.metrics_json
                            jobs[job.job_id] = {
                                "job_id": job.job_id, "model": job.config.model_type,
                                "state": pb.State.Name(job.state), "worker_id": job.worker_id,
                                "progress": job.progress, "epoch": job.epoch,
                                "metrics": metrics, "error": job.error,
                            }
                        elif kind in ("worker_registered", "worker_dead"):
                            workers[event.worker_id] = "ALIVE" if kind == "worker_registered" else "DEAD"
                        elif kind.startswith("reassignments:"):
                            count = int(kind.split(":", 1)[1])
                        elif kind == "snapshot_end":
                            self.connected = True
                            delay = 0.5
                        if self.connected:
                            self.jobs, self.workers, self.reassignments = jobs, workers, count
                            self.notify()
                except grpc.aio.AioRpcError:
                    pass
                self.connected = False
                self.notify()
                await asyncio.sleep(random.uniform(delay / 2, delay))
                delay = min(delay * 2, 8)


@asynccontextmanager
async def lifespan(app):
    feed = app.state.feed = Feed()
    consumer = asyncio.create_task(feed.consume())
    try:
        yield
    finally:
        consumer.cancel()
        with suppress(asyncio.CancelledError):
            await consumer


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/events")
async def events():
    feed = app.state.feed

    async def stream():
        listener = asyncio.Event()
        feed.listeners.add(listener)
        listener.set()
        try:
            while True:
                try:
                    await asyncio.wait_for(listener.wait(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                listener.clear()
                yield f"data: {json.dumps(feed.snapshot())}\n\n"
        finally:
            feed.listeners.discard(listener)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})

