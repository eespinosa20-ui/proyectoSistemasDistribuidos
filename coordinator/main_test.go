package main

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	pb "training/proto"
)

func TestReassignmentAndPersistence(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "queue.json")
	c, err := openCoordinator(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &pb.JobConfig{ModelType: "mlp", DataVersion: "mock-v1"}
	job, err := c.SubmitJob(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"first", "second"} {
		if _, err := c.RegisterWorker(ctx, &pb.Worker{WorkerId: name}); err != nil {
			t.Fatal(err)
		}
	}
	pull := func(name string) *pb.Job {
		t.Helper()
		next, err := c.RequestJob(ctx, &pb.Pull{WorkerId: name, ModelTypes: []string{"mlp"}, RequestId: "request"})
		if err != nil {
			t.Fatal(err)
		}
		if next.JobId != job.JobId || next.State != pb.State_RUNNING {
			t.Fatalf("unexpected assignment: %v", next)
		}
		return next
	}
	first := pull("first")
	if again := pull("first"); again.Attempt != first.Attempt {
		t.Fatal("pull retry allocated another attempt")
	}
	c.workers["first"].seen = time.Now().Add(-heartbeatLimit)
	second := pull("second")
	if second.Attempt != first.Attempt+1 {
		t.Fatal("reassignment did not advance attempt")
	}
	if _, ok := c.workers["first"]; ok {
		t.Fatal("expired worker survived")
	}
	stale := &pb.Update{JobId: job.JobId, Attempt: first.Attempt, WorkerId: first.WorkerId}
	if _, err := c.CompleteJob(ctx, stale); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("stale completion returned %v", err)
	}
	recovered, err := openCoordinator(path)
	if err != nil {
		t.Fatal(err)
	}
	existing, err := recovered.SubmitJob(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if existing.JobId != job.JobId || existing.State != pb.State_REASSIGNED ||
		existing.Attempt != second.Attempt {
		t.Fatalf("unexpected recovered job: %v", existing)
	}
}

func TestGridAndTimeout(t *testing.T) {
	ctx := context.Background()
	c, err := openCoordinator(filepath.Join(t.TempDir(), "queue.json"))
	if err != nil {
		t.Fatal(err)
	}
	base := &pb.JobConfig{ModelType: "linear_regression", DataVersion: "mock-v1"}
	grid := &pb.Grid{Base: base, Axes: map[string]*pb.Values{
		"alpha":   {Json: []string{"0.1", "1", "0.1"}},
		"variant": {Json: []string{"\"ridge\"", "\"lasso\""}},
	}}
	result, err := c.SubmitGridSearch(ctx, grid)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Jobs) != 6 || len(c.order) != 4 {
		t.Fatalf("unexpected grid: %v", result)
	}
	if _, err := c.RegisterWorker(ctx, &pb.Worker{WorkerId: "healthy"}); err != nil {
		t.Fatal(err)
	}
	job, err := c.RequestJob(ctx, &pb.Pull{WorkerId: "healthy", ModelTypes: []string{base.ModelType}, RequestId: "1"})
	if err != nil {
		t.Fatal(err)
	}
	atDeadline := time.UnixMilli(job.DeadlineUnixMs)
	c.workers["healthy"].seen = atDeadline
	if err := c.expire(atDeadline); err != nil {
		t.Fatal(err)
	}
	if c.jobs[job.JobId].State != pb.State_REASSIGNED || c.workers["healthy"] == nil {
		t.Fatal("deadline should reassign the job and preserve the worker")
	}
}
