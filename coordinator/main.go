package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	pb "training/proto"
)

var deadlines = map[string]time.Duration{
	"linear_regression": 30 * time.Second, "mlp": 180 * time.Second, "cnn": 600 * time.Second,
}

const heartbeatLimit = 6 * time.Second

type worker struct {
	seen        time.Time
	lastRequest string
	lastJob     *pb.Job
}

type coordinator struct {
	pb.UnimplementedTrainingServer
	mu        sync.Mutex
	path      string
	jobs      map[string]*pb.Job
	order     []string
	workers   map[string]*worker
	listeners map[chan *pb.Event]struct{}
}

func openCoordinator(path string) (*coordinator, error) {
	c := &coordinator{path: path, jobs: map[string]*pb.Job{}, workers: map[string]*worker{},
		listeners: map[chan *pb.Event]struct{}{}}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	var jobs []*pb.Job
	if err := json.Unmarshal(data, &jobs); err != nil {
		return nil, err
	}
	var recovered []*pb.Job
	for _, job := range jobs {
		c.jobs[job.JobId] = job
		c.order = append(c.order, job.JobId)
		if job.State == pb.State_RUNNING {
			recovered = append(recovered, reassigned(job))
		}
	}
	if len(recovered) > 0 {
		err = c.commit(recovered...)
	}
	return c, err
}

func (c *coordinator) commit(changes ...*pb.Job) error {
	replacements := make(map[string]*pb.Job, len(changes))
	order := append([]string(nil), c.order...)
	for _, job := range changes {
		if _, exists := c.jobs[job.JobId]; !exists && replacements[job.JobId] == nil {
			order = append(order, job.JobId)
		}
		replacements[job.JobId] = job
	}
	jobs := make([]*pb.Job, 0, len(order))
	for _, id := range order {
		job := replacements[id]
		if job == nil {
			job = c.jobs[id]
		}
		jobs = append(jobs, job)
	}
	if err := save(c.path, jobs); err != nil {
		log.Print(err)
		return status.Error(codes.Internal, "could not persist queue")
	}
	c.order = order
	for id, job := range replacements {
		c.jobs[id] = job
		c.emit(&pb.Event{Kind: "job", Job: job, WorkerId: job.WorkerId})
	}
	return nil
}

func save(path string, jobs []*pb.Job) error {
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0755); err != nil {
		return err
	}
	file, err := os.CreateTemp(directory, ".queue-*")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	err = json.NewEncoder(file).Encode(jobs)
	if err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(file.Name(), path)
}

func clone(job *pb.Job) *pb.Job { return proto.Clone(job).(*pb.Job) }

func identify(config *pb.JobConfig) (*pb.Job, error) {
	if config == nil || deadlines[config.ModelType] == 0 {
		return nil, status.Error(codes.InvalidArgument, "unsupported model_type")
	}
	config = proto.Clone(config).(*pb.JobConfig)
	if config.DataVersion == "" {
		return nil, status.Error(codes.InvalidArgument, "data_version is required")
	}
	params := map[string]json.RawMessage{}
	for name, raw := range config.Hyperparameters {
		var value any
		decoder := json.NewDecoder(strings.NewReader(raw))
		decoder.UseNumber()
		if decoder.Decode(&value) != nil || decoder.Decode(new(any)) != io.EOF {
			return nil, status.Errorf(codes.InvalidArgument, "invalid JSON hyperparameter %q", name)
		}
		canonical, err := json.Marshal(value)
		if err != nil {
			return nil, status.Error(codes.InvalidArgument, err.Error())
		}
		params[name] = canonical
		config.Hyperparameters[name] = string(canonical)
	}
	encoded, err := json.Marshal(map[string]any{"model_type": config.ModelType,
		"hyperparameters": params, "data_version": config.DataVersion, "seed": config.Seed})
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	return &pb.Job{JobId: fmt.Sprintf("%x", sha256.Sum256(encoded)), Config: config}, nil
}

func (c *coordinator) submit(configs []*pb.JobConfig) ([]*pb.Job, error) {
	jobs := make([]*pb.Job, 0, len(configs))
	for _, config := range configs {
		job, err := identify(config)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, job)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	var changes []*pb.Job
	seen := map[string]*pb.Job{}
	for i, job := range jobs {
		switch {
		case c.jobs[job.JobId] != nil:
			jobs[i] = c.jobs[job.JobId]
		case seen[job.JobId] != nil:
			jobs[i] = seen[job.JobId]
		default:
			changes = append(changes, job)
			seen[job.JobId] = job
		}
	}
	if len(changes) > 0 {
		if err := c.commit(changes...); err != nil {
			return nil, err
		}
	}
	return jobs, nil
}

func (c *coordinator) SubmitJob(_ context.Context, config *pb.JobConfig) (*pb.Job, error) {
	jobs, err := c.submit([]*pb.JobConfig{config})
	if err != nil {
		return nil, err
	}
	return jobs[0], nil
}

func (c *coordinator) SubmitGridSearch(_ context.Context, grid *pb.Grid) (*pb.Jobs, error) {
	if _, err := identify(grid.Base); err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(grid.Axes))
	for key := range grid.Axes {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	configs := []*pb.JobConfig{grid.Base}
	for _, key := range keys {
		values := grid.Axes[key].GetJson()
		if len(values) == 0 || len(configs) > 10000/len(values) {
			return nil, status.Error(codes.InvalidArgument, "grid must contain 1 to 10000 combinations")
		}
		next := make([]*pb.JobConfig, 0, len(configs)*len(values))
		for _, config := range configs {
			for _, value := range values {
				variant := proto.Clone(config).(*pb.JobConfig)
				if variant.Hyperparameters == nil {
					variant.Hyperparameters = map[string]string{}
				}
				variant.Hyperparameters[key] = value
				next = append(next, variant)
			}
		}
		configs = next
	}
	jobs, err := c.submit(configs)
	return &pb.Jobs{Jobs: jobs}, err
}

func (c *coordinator) GetJobStatus(_ context.Context, ref *pb.JobRef) (*pb.Job, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if job := c.jobs[ref.JobId]; job != nil {
		return job, nil
	}
	return nil, status.Error(codes.NotFound, "job not found")
}

func (c *coordinator) RegisterWorker(_ context.Context, req *pb.Worker) (*pb.Empty, error) {
	if req.WorkerId == "" {
		return nil, status.Error(codes.InvalidArgument, "worker_id is required")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now()
	if err := c.expire(now); err != nil {
		return nil, err
	}
	if c.workers[req.WorkerId] == nil {
		c.workers[req.WorkerId] = &worker{}
		c.emit(&pb.Event{Kind: "worker_registered", WorkerId: req.WorkerId})
	}
	c.workers[req.WorkerId].seen = now
	return &pb.Empty{}, nil
}

func (c *coordinator) Heartbeat(_ context.Context, req *pb.Worker) (*pb.Empty, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now()
	w := c.workers[req.WorkerId]
	if w == nil || now.Sub(w.seen) >= heartbeatLimit {
		return nil, status.Error(codes.NotFound, "worker must register")
	}
	w.seen = now
	return &pb.Empty{}, nil
}

func (c *coordinator) RequestJob(_ context.Context, req *pb.Pull) (*pb.Job, error) {
	if req.RequestId == "" {
		return nil, status.Error(codes.InvalidArgument, "request_id is required")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now()
	if err := c.expire(now); err != nil {
		return nil, err
	}
	w := c.workers[req.WorkerId]
	if w == nil {
		return nil, status.Error(codes.NotFound, "worker must register")
	}
	if w.lastRequest == req.RequestId {
		if w.lastJob.JobId == "" {
			return w.lastJob, nil
		}
		current := c.jobs[w.lastJob.JobId]
		if current.Attempt == w.lastJob.Attempt && current.State == pb.State_RUNNING {
			return current, nil
		}
		return &pb.Job{}, nil
	}
	models := map[string]bool{}
	for _, model := range req.ModelTypes {
		models[model] = true
	}
	for _, id := range c.order {
		job := c.jobs[id]
		if (job.State != pb.State_PENDING && job.State != pb.State_REASSIGNED) || !models[job.Config.ModelType] {
			continue
		}
		next := clone(job)
		next.State, next.WorkerId = pb.State_RUNNING, req.WorkerId
		next.Attempt++
		next.DeadlineUnixMs = now.Add(deadlines[job.Config.ModelType]).UnixMilli()
		if err := c.commit(next); err != nil {
			return nil, err
		}
		w.lastRequest, w.lastJob = req.RequestId, next
		return next, nil
	}
	w.lastRequest, w.lastJob = req.RequestId, &pb.Job{}
	return w.lastJob, nil
}

func (c *coordinator) update(req *pb.Update, complete bool) (*pb.Empty, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now()
	job := c.jobs[req.JobId]
	if job == nil {
		return nil, status.Error(codes.NotFound, "job not found")
	}
	if job.WorkerId == req.WorkerId && job.Attempt == req.Attempt && complete &&
		(job.State == pb.State_COMPLETED || job.State == pb.State_FAILED) {
		return &pb.Empty{}, nil
	}
	w := c.workers[req.WorkerId]
	if job.State != pb.State_RUNNING || job.WorkerId != req.WorkerId || job.Attempt != req.Attempt ||
		w == nil || now.Sub(w.seen) >= heartbeatLimit || now.UnixMilli() >= job.DeadlineUnixMs {
		return nil, status.Error(codes.FailedPrecondition, "assignment expired")
	}
	next := clone(job)
	if complete {
		next.DeadlineUnixMs = 0
		if req.Error != "" {
			next.State, next.Error = pb.State_FAILED, req.Error
		} else {
			next.State, next.Progress = pb.State_COMPLETED, 1
			next.MetricsJson, next.ModelPath = req.MetricsJson, req.ModelPath
		}
	} else {
		if !(req.Progress >= 0 && req.Progress <= 1) {
			return nil, status.Error(codes.InvalidArgument, "progress outside [0,1]")
		}
		next.Epoch, next.Progress = req.Epoch, req.Progress
	}
	if err := c.commit(next); err != nil {
		return nil, err
	}
	return &pb.Empty{}, nil
}

func (c *coordinator) ReportProgress(_ context.Context, req *pb.Update) (*pb.Empty, error) {
	return c.update(req, false)
}

func (c *coordinator) CompleteJob(_ context.Context, req *pb.Update) (*pb.Empty, error) {
	return c.update(req, true)
}

func reassigned(job *pb.Job) *pb.Job {
	next := clone(job)
	next.State, next.WorkerId, next.DeadlineUnixMs = pb.State_REASSIGNED, "", 0
	return next
}

func (c *coordinator) expire(now time.Time) error {
	dead := map[string]bool{}
	for id, w := range c.workers {
		if now.Sub(w.seen) >= heartbeatLimit {
			dead[id] = true
		}
	}
	var changes []*pb.Job
	for _, id := range c.order {
		job := c.jobs[id]
		if job.State == pb.State_RUNNING && (dead[job.WorkerId] || now.UnixMilli() >= job.DeadlineUnixMs) {
			changes = append(changes, reassigned(job))
		}
	}
	if len(changes) > 0 {
		if err := c.commit(changes...); err != nil {
			return err
		}
	}
	for id := range dead {
		delete(c.workers, id)
		c.emit(&pb.Event{Kind: "worker_dead", WorkerId: id})
	}
	return nil
}

func (c *coordinator) emit(event *pb.Event) {
	for listener := range c.listeners {
		select {
		case listener <- event:
		default:
			close(listener)
			delete(c.listeners, listener)
		}
	}
}

func (c *coordinator) StreamEvents(_ *pb.Empty, stream grpc.ServerStreamingServer[pb.Event]) error {
	listener := make(chan *pb.Event, 128)
	c.mu.Lock()
	snapshot := make([]*pb.Event, 0, len(c.order)+len(c.workers))
	for _, id := range c.order {
		snapshot = append(snapshot, &pb.Event{Kind: "job", Job: c.jobs[id]})
	}
	for id := range c.workers {
		snapshot = append(snapshot, &pb.Event{Kind: "worker_registered", WorkerId: id})
	}
	c.listeners[listener] = struct{}{}
	c.mu.Unlock()
	defer func() { c.mu.Lock(); delete(c.listeners, listener); c.mu.Unlock() }()
	for _, event := range snapshot {
		if err := stream.Send(event); err != nil {
			return err
		}
	}
	for {
		select {
		case <-stream.Context().Done():
			return stream.Context().Err()
		case event, ok := <-listener:
			if !ok {
				return status.Error(codes.ResourceExhausted, "event consumer fell behind")
			}
			if err := stream.Send(event); err != nil {
				return err
			}
		}
	}
}

func main() {
	if len(os.Args) == 2 && os.Args[1] == "health" {
		conn, err := grpc.NewClient("127.0.0.1:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			log.Fatal(err)
		}
		defer conn.Close()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		result, err := healthpb.NewHealthClient(conn).Check(ctx, &healthpb.HealthCheckRequest{})
		if err != nil {
			log.Fatal(err)
		}
		if result.Status != healthpb.HealthCheckResponse_SERVING {
			log.Fatal(result.Status)
		}
		return
	}
	path := os.Getenv("QUEUE_PATH")
	if path == "" {
		path = "queue.json"
	}
	c, err := openCoordinator(path)
	if err != nil {
		log.Fatal(err)
	}
	go func() {
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for now := range ticker.C {
			c.mu.Lock()
			_ = c.expire(now)
			c.mu.Unlock()
		}
	}()
	listen, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatal(err)
	}
	server := grpc.NewServer()
	pb.RegisterTrainingServer(server, c)
	healthServer := health.NewServer()
	healthpb.RegisterHealthServer(server, healthServer)
	log.Print("coordinator listening on :50051")
	log.Fatal(server.Serve(listen))
}
