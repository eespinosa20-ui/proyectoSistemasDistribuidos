import hashlib
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc

import main
import training_pb2 as pb
import training_pb2_grpc as rpc


class CoordinatorTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "queue.json"
        self.coordinator = main.Coordinator(self.path)
        server = grpc.server(ThreadPoolExecutor(max_workers=8))
        rpc.add_TrainingServicer_to_server(self.coordinator, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        self.addCleanup(lambda: server.stop(0).wait())
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.addCleanup(channel.close)
        self.stub = rpc.TrainingStub(channel)
        self.config = pb.JobConfig(model_type="linear_regression", data_version="mock-v1", seed=42,
                                   hyperparameters={"alpha": " 0.1 "})

    def pull(self, worker_id, request_id="1"):
        self.stub.RegisterWorker(pb.Worker(worker_id=worker_id), timeout=2)
        return self.stub.RequestJob(pb.Pull(worker_id=worker_id, request_id=request_id,
                                            model_types=["linear_regression"]), timeout=2)

    def update(self, job):
        return pb.Update(job_id=job.job_id, worker_id=job.worker_id, attempt=job.attempt,
                         progress=1, metrics_json='{"val": {}, "test": {}}')

    def assert_rpc_code(self, code, call, request):
        with self.assertRaises(grpc.RpcError) as error:
            call(request, timeout=2)
        self.assertEqual(error.exception.code(), code)

    def test_idempotency_with_concurrent_submissions_and_completion(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(lambda _: self.stub.SubmitJob(self.config, timeout=2), range(16)))
        expected = hashlib.sha256(json.dumps({
            "model_type": "linear_regression", "data_version": "mock-v1",
            "hyperparameters": {"alpha": 0.1}, "seed": 42,
        }, sort_keys=True).encode()).hexdigest()
        self.assertEqual({job.job_id for job in jobs}, {expected})
        self.assertEqual(len(self.coordinator.jobs), 1)
        assignment = self.pull("first")
        self.assertEqual(self.pull("first"), assignment)
        self.stub.CompleteJob(self.update(assignment), timeout=2)
        self.stub.CompleteJob(self.update(assignment), timeout=2)
        completed = self.stub.SubmitJob(self.config, timeout=2)
        self.assertEqual(completed.state, pb.COMPLETED)
        self.assertEqual(main.Coordinator(self.path).jobs[expected], completed)

    def test_invalid_model(self):
        self.config.model_type = "random_forest"
        self.assert_rpc_code(grpc.StatusCode.INVALID_ARGUMENT, self.stub.SubmitJob, self.config)
        self.assert_rpc_code(grpc.StatusCode.INVALID_ARGUMENT, self.stub.SubmitGridSearch,
                             pb.Grid(base=self.config))
        self.assertFalse(self.coordinator.jobs)

    def test_timeout_rejects_late_result(self):
        self.stub.SubmitJob(self.config, timeout=2)
        assignment = self.pull("first")
        self.coordinator.expire(time.monotonic(), assignment.deadline_unix_ms / 1000 + 0.001)
        next_job = self.pull("second")
        self.assertTrue(self.coordinator.workers["first"].alive)
        self.assertEqual(next_job.job_id, assignment.job_id)
        self.assertEqual(next_job.attempt, 2)
        self.assertEqual(self.coordinator.reassignments, 1)
        self.assert_rpc_code(grpc.StatusCode.FAILED_PRECONDITION, self.stub.CompleteJob,
                             self.update(assignment))

    def test_heartbeat_failure_requeues_and_increments_count(self):
        self.stub.SubmitJob(self.config, timeout=2)
        assignment = self.pull("first")
        self.coordinator.expire(self.coordinator.workers["first"].seen + 6.01, time.time())
        self.assertFalse(self.coordinator.workers["first"].alive)
        self.assertEqual(self.coordinator.jobs[assignment.job_id].state, pb.PENDING)
        self.assertEqual(self.coordinator.reassignments, 1)
        self.assert_rpc_code(grpc.StatusCode.NOT_FOUND, self.stub.Heartbeat,
                             pb.Worker(worker_id="first"))
        self.assertEqual(self.pull("second").attempt, 2)
        self.assert_rpc_code(grpc.StatusCode.FAILED_PRECONDITION, self.stub.ReportProgress,
                             self.update(assignment))

    def test_restarts_with_legacy_queue_and_preserves_id(self):
        self.path.write_text(json.dumps([{
            "job_id": "legacy-id", "config": {"model_type": "linear_regression",
                "hyperparameters": {"alpha": "0.1"}, "data_version": "mock-v1", "seed": 42},
            "state": 1, "attempt": 1, "worker_id": "first", "deadline_unix_ms": 123,
        }]), encoding="utf-8")
        recovered = main.Coordinator(self.path)
        self.assertEqual(recovered.jobs["legacy-id"].state, pb.PENDING)
        self.assertEqual(recovered.reassignments, 1)
        self.assertEqual(recovered.submit([self.config], None)[0].job_id, "legacy-id")
        self.assertEqual(main.Coordinator(self.path).reassignments, 1)

    def test_grid_deduplicates_overlapping_combinations(self):
        grid = pb.Grid(base=self.config, axes={"alpha": pb.Values(json=["0.1", " 0.1", "1"])})
        jobs = self.stub.SubmitGridSearch(grid, timeout=2).jobs
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0].job_id, jobs[1].job_id)
        self.assertEqual(len(self.coordinator.jobs), 2)


if __name__ == "__main__":
    unittest.main()

