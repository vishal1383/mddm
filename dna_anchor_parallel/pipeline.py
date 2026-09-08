"""Bounded GPU RPC; one model per GPU, several independent CPU clients.

No batch merging: every request retains the original batch shape, precision and
token proposal implementation. CPU workers have independent seeded torch RNGs.
"""
import json
import os
import queue
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

LEGACY = Path(__file__).resolve().parents[1] / 'dna_anchor_census'
sys.path.insert(0, str(LEGACY))


def cpu_setup(threads):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


class ForwardProxy:
    def __init__(self, client, requests, replies, fatal, timeout=600):
        self.client, self.requests, self.replies = client, requests, replies
        self.fatal, self.timeout = fatal, timeout
        self.reset(-1)

    def reset(self, index):
        self.index, self.calls, self.sequences, self.hidden = index, 0, 0, None

    def __call__(self, **kwargs):
        root = self.calls == 0
        self.calls += 1
        self.sequences += kwargs['input_ids'].shape[0]
        message = (self.client, self.index, self.calls, root, kwargs)
        deadline = time.monotonic() + self.timeout
        while True:
            if self.fatal.is_set():
                raise RuntimeError('GPU pipeline failed; see worker error journal')
            if time.monotonic() > deadline:
                raise TimeoutError(f'GPU response timed out for {self.index}')
            try:
                self.requests.put(message, timeout=1)
                break
            except queue.Full:
                continue
        while True:
            if self.fatal.is_set():
                raise RuntimeError('GPU pipeline failed; see worker error journal')
            if time.monotonic() > deadline:
                raise TimeoutError(f'GPU response timed out for {self.index}')
            try:
                index, call, logits, hidden = self.replies.get(timeout=1)
                break
            except queue.Empty:
                continue
        assert index == self.index and call == self.calls, 'RPC crossed context boundaries'
        if root:
            self.hidden = hidden
        return SimpleNamespace(logits=logits)


def serve(rank, device, model_path, requests, replies, events, fatal, shutdown,
          journal_path, fake=False):
    """GPU stays resident until all CPU clients have released shared tensors."""
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    calls = sequences = 0
    idle_seconds = busy_seconds = 0.
    last_stats = time.monotonic()
    try:
        if not fake:
            from transformers import AutoModelForMaskedLM
            from run import require_a100
            name = require_a100()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model, loading = AutoModelForMaskedLM.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True,
                torch_dtype=torch.float32, output_loading_info=True)
            assert not any(loading[k] for k in
                           ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
            model = model.eval().requires_grad_(False).to('cuda:0')
            uuid = str(torch.cuda.get_device_properties(0).uuid)
        else:
            name, uuid = 'FAKE_CPU_TEST_ONLY', f'fake-{rank}'
        events.put(dict(event='gpu_ready', rank=rank, gpu=name, uuid=uuid,
                        visible_device=device))
        with Path(journal_path).open('x', buffering=1) as journal:
            def log(kind, **fields):
                journal.write(json.dumps(dict(event=kind, unix=time.time(), rank=rank, **fields)) + '\n')
            def stats():
                return dict(rank=rank, physical_calls=calls, sequence_forwards=sequences,
                            queue_empty_seconds=idle_seconds, inference_and_transfer_seconds=busy_seconds)
            while not shutdown.is_set() and not fatal.is_set():
                tick = time.monotonic()
                try:
                    client, index, call, root, kwargs = requests.get(timeout=.2)
                except queue.Empty:
                    idle_seconds += time.monotonic() - tick
                    continue
                idle_seconds += time.monotonic() - tick
                tick = time.monotonic()
                n = kwargs['input_ids'].shape[0]
                log('forward_started', index=index, client=client, context_call=call, sequences=n)
                with torch.inference_mode():
                    if fake:
                        logits = kwargs['input_ids'].float().unsqueeze(-1).repeat(1, 1, 3)
                        hidden = kwargs['input_ids'][0].float().unsqueeze(-1) if root else None
                    else:
                        output = model(**{k: v.to('cuda:0') for k, v in kwargs.items()},
                                       output_hidden_states=root)
                        logits = output.logits.float().cpu()
                        hidden = output.hidden_states[-1][0].float().cpu() if root else None
                        del output
                # Queue transfers use torch shared-memory handles, not pickled 180-MB tensors.
                # Each CPU client has at most one outstanding request/response.
                replies[client].put((index, call, logits, hidden), timeout=60)
                calls += 1
                sequences += n
                busy_seconds += time.monotonic() - tick
                log('forward_finished', index=index, client=client, context_call=call, sequences=n)
                if time.monotonic() - last_stats >= 30:
                    events.put(dict(event='gpu_statistics', **stats()))
                    journal.flush(); os.fsync(journal.fileno())
                    last_stats = time.monotonic()
            log('gpu_stopped', **{k: v for k, v in stats().items() if k != 'rank'})
            journal.flush(); os.fsync(journal.fileno())
        events.put(dict(event='gpu_stopped', **stats()))
        # Shutdown is requested only after clients have consumed their replies.
        # Do not let torch IPC feeder cleanup keep a finished GPU process alive.
        for reply in replies:
            reply.cancel_join_thread()
    except BaseException:
        events.put(dict(event='error', role='gpu', rank=rank, traceback=traceback.format_exc()))
        fatal.set()
        raise


def sampler_from_source(model_path):
    """Load native sampling function without allocating another copy of weights."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    config = json.loads((Path(model_path) / 'config.json').read_text())
    cls = get_class_from_dynamic_module(config['auto_map']['AutoModelForMaskedLM'],
                                       str(model_path), local_files_only=True)
    return sys.modules[cls.__module__]._sample_tokens
