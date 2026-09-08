"""Four-A100, CPU-pipelined continuation of an existing D3LM census cache."""
import argparse
import collections
import fcntl
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
# Keep sibling imports ahead of legacy/run.py; worker imports that module by name.
sys.path.insert(0, str(HERE.parent / 'dna_anchor_census'))
import data
from cache import validate_contract, completed_indices
from pipeline import serve, sampler_from_source
from worker import work


def telemetry(uuids, output, shutdown, samples):
    fields = 'timestamp,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw'
    with output.open('x', buffering=1) as f:
        f.write(fields + '\n')
        while not shutdown.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '-i', ','.join(uuids),
                    '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=10, check=True)
                f.write(result.stdout)
                for line in result.stdout.strip().splitlines():
                    parts = [s.strip() for s in line.split(',')]
                    samples.append((time.time(), parts[1], float(parts[3])))
            except Exception as exc:
                f.write('# telemetry_error ' + repr(exc) + '\n')
            shutdown.wait(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--clients-per-gpu', type=int, default=4)
    ap.add_argument('--cpu-threads', type=int, default=2)
    ap.add_argument('--max-new', type=int, help='Bounded performance diagnostic; does not change corpus contract')
    ap.add_argument('--scratch-parent', type=Path, default=os.environ.get('TMPDIR', '/tmp'))
    args = ap.parse_args()
    assert 1 <= args.clients_per_gpu <= 8 and args.cpu_threads > 0
    assert args.max_new is None or args.max_new > 0
    state = args.state.resolve()
    lock = (state / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 4, 'Exactly four GPUs required'
    names = [torch.cuda.get_device_name(i) for i in range(4)]
    assert all('A100' in name.upper() for name in names), names
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    assert len(visible) == 4 and len(set(visible)) == 4, visible
    records, inventory = data.records(state / 'assets/dataset/sequence.csv', 'all_species')
    contract = validate_contract(state, records)
    complete = completed_indices(state, records)
    assert all(i in complete for i in [1, 2, 3, 4]), 'Four original reference caches required for parity'
    remaining = [r for r in records if r['index'] not in complete]
    if args.max_new is not None:
        remaining = remaining[:args.max_new]
    invocation = f'{os.environ.get("SLURM_JOB_ID", "local")}-{time.time_ns()}'
    run_dir = state / 'parallel_runs' / invocation
    run_dir.mkdir(parents=True)
    data.atomic(run_dir / 'execution_contract.json', dict(
        legacy_contract_sha256=data.sha(state / 'contract.json'),
        execution_source_sha256={p.name: data.sha(p) for p in HERE.iterdir() if p.suffix in {'.py', '.sbatch'}},
        initial_completed=len(complete), full_corpus=len(records), queued_contexts=len(remaining),
        gpus=4, clients_per_gpu=args.clients_per_gpu, cpu_threads_per_client=args.cpu_threads,
        native_batch_size=32, batch_merging=False, precision='float32; TF32 disabled',
        cpu_native_sampler=True, global_dynamic_work_queue=True,
        parity_indices=[1, 2, 3, 4], numerical_atol=2e-5, numerical_rtol=1e-6,
        exact_discrete_parity_required=True, names=names, visible_devices=visible,
        new_model_training_steps=0))
    # Populate the HF custom-code cache once before concurrent imports.
    model_path = state / 'assets/model'
    sampler_from_source(model_path)
    ctx = mp.get_context('spawn')
    fatal, stop, shutdown, parity_gate = (ctx.Event() for _ in range(4))
    def request_stop(signum, frame):
        stop.set()
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    events = ctx.Queue()
    clients_count = args.clients_per_gpu * 4
    tasks = ctx.Queue(maxsize=clients_count * 2)
    servers, clients = [], []
    ready, parity = {}, set()
    gpu_stats, stopped_clients = {}, set()
    samples = collections.deque(maxlen=4 * 720)  # one hour of per-GPU samples
    telemetry_stop = threading.Event()
    telemetry_thread = None
    started = time.monotonic()
    production_started = None
    completion_times = collections.deque(maxlen=400)
    completed_new, dispatched, sentinels = 0, 0, 0
    last_status = 0.
    interrupted_at = None
    journal = (run_dir / 'events.jsonl').open('x', buffering=1)
    def log(event):
        event = dict(unix=time.time(), **event)
        journal.write(json.dumps(event) + '\n')
        if event['event'] in {'context_complete', 'error', 'parity_pass'}:
            journal.flush(); os.fsync(journal.fileno())
    def status(phase):
        now = time.monotonic()
        rate = None
        if len(completion_times) >= 32:
            rate = (len(completion_times)-1) * 3600 / (completion_times[-1]-completion_times[0])
        cutoff = time.time()-180
        snapshot = list(samples)
        util = {}
        for rank, item in ready.items():
            values = [v for t, uuid, v in snapshot if t >= cutoff and uuid == item['uuid']]
            util[rank] = sum(values)/len(values) if values else None
        result = dict(status=phase, slurm_job_id=os.environ.get('SLURM_JOB_ID'), run_directory=str(run_dir),
                      completed=len(complete), completed_this_invocation=completed_new,
                      expected=len(records), elapsed_seconds=now-started,
                      production_seconds=None if production_started is None else now-production_started,
                      recent_contexts_per_hour_all_gpus=rate,
                      projected_remaining_hours=None if not rate else (len(records)-len(complete))/rate,
                      gpu_utilization_percent_last_180_seconds=util, gpu_statistics=gpu_stats,
                      gpu_identities=ready, parity_passed_ranks=sorted(parity), new_model_training_steps=0)
        data.atomic(run_dir / 'status.json', result)
        data.atomic(state / 'status.json', result)
        print(json.dumps(result), flush=True)
    try:
        for rank in range(4):
            requests = ctx.Queue(maxsize=args.clients_per_gpu)
            replies = [ctx.Queue(maxsize=1) for _ in range(args.clients_per_gpu)]
            server = ctx.Process(target=serve, args=(rank, visible[rank], model_path, requests, replies,
                events, fatal, shutdown, run_dir / f'forwards-rank{rank}.jsonl'), name=f'dna-gpu-{rank}')
            server.start(); servers.append(server)
            for client in range(args.clients_per_gpu):
                p = ctx.Process(target=work, args=(client, rank, model_path, state, run_dir, tasks,
                    requests, replies[client], events, stop, fatal, args.cpu_threads,
                    records[rank+1] if client == 0 else None, parity_gate, args.scratch_parent),
                    name=f'dna-cpu-{rank}-{client}')
                p.start(); clients.append(p)
        while len(stopped_clients) < clients_count:
            # Dynamic queue balances all clients/GPUs; no rank can finish a fixed shard early.
            if parity_gate.is_set() and not stop.is_set() and not fatal.is_set():
                while dispatched < len(remaining):
                    try:
                        tasks.put_nowait(remaining[dispatched]); dispatched += 1
                    except queue.Full:
                        break
                if dispatched == len(remaining):
                    while sentinels < clients_count:
                        try:
                            tasks.put_nowait(None); sentinels += 1
                        except queue.Full:
                            break
            try:
                event = events.get(timeout=.2)
                log(event)
                kind = event['event']
                if kind == 'gpu_ready':
                    assert event['gpu'] == contract['device_name'], 'GPU model differs from original parity contract'
                    assert event['uuid'] not in [v['uuid'] for v in ready.values()], 'Duplicate GPU assignment'
                    ready[event['rank']] = event
                    if len(ready) == 4:
                        telemetry_thread = threading.Thread(target=telemetry, args=(
                            [ready[i]['uuid'] for i in range(4)], run_dir / 'gpu_utilization.csv',
                            telemetry_stop, samples), daemon=True)
                        telemetry_thread.start()
                elif kind == 'parity_pass':
                    parity.add(event['rank'])
                    if len(parity) == 4:
                        production_started = time.monotonic()
                        parity_gate.set()
                        log(dict(event='all_four_parity_pass_production_started'))
                elif kind == 'context_complete':
                    assert event['index'] not in complete, 'Duplicate completion'
                    complete.add(event['index']); completed_new += 1
                    completion_times.append(time.monotonic())
                elif kind == 'gpu_statistics':
                    gpu_stats[event['rank']] = event
                elif kind == 'client_stopped':
                    stopped_clients.add((event['rank'], event['client']))
                elif kind == 'error':
                    fatal.set(); stop.set()
                    print(json.dumps(event), flush=True)
            except queue.Empty:
                pass
            for process in servers + clients:
                if process.exitcode not in (None, 0):
                    fatal.set(); stop.set()
                    log(dict(event='process_failed', process=process.name, exitcode=process.exitcode))
            if fatal.is_set():
                raise RuntimeError('Worker failed; complete archives remain resumable; inspect events.jsonl')
            if stop.is_set():
                if interrupted_at is None:
                    interrupted_at = time.monotonic()
                if time.monotonic()-interrupted_at > 100:
                    raise TimeoutError('Drain exceeded 100 seconds; incomplete scratch retained')
            if production_started is None and time.monotonic()-started > 600:
                raise TimeoutError('Startup/parity did not finish within ten minutes')
            if time.monotonic()-last_status > 30:
                status('DRAINING' if stop.is_set() else ('RUNNING_PARALLEL' if parity_gate.is_set() else 'PREFLIGHT_PARITY'))
                last_status = time.monotonic()
        for process in clients:
            process.join(timeout=5)
            assert process.exitcode == 0, process.name
        shutdown.set()
        for process in servers:
            process.join(timeout=15)
            assert process.exitcode == 0, process.name
        phase = 'COMPLETE_CENSUS' if len(complete) == len(records) else (
            'COMPLETE_LIMITED_DIAGNOSTIC' if args.max_new and completed_new == len(remaining) else 'INTERRUPTED_RESUMABLE')
        status(phase)
    except BaseException as exc:
        fatal.set(); stop.set()
        log(dict(event='coordinator_failed', error=repr(exc)))
        status('FAILED_RECOVERABLE')
        raise
    finally:
        telemetry_stop.set(); shutdown.set()
        # Only processes created by this invocation; never touch unrelated research jobs.
        for process in clients + servers:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate(); process.join(timeout=2)
                if process.is_alive():
                    process.kill(); process.join(timeout=2)
        if telemetry_thread is not None:
            telemetry_thread.join(timeout=12)
        journal.close(); lock.close()
        # On an interrupted run, queued records are recoverable from the source manifest.
        tasks.cancel_join_thread()


if __name__ == '__main__':
    main()
