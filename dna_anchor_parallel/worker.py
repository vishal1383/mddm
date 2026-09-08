import os
import queue
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

from pipeline import ForwardProxy, cpu_setup, sampler_from_source, LEGACY
sys.path.insert(0, str(LEGACY))


def work(client, rank, model_path, state, run_dir, tasks, requests, replies,
         events, stop, fatal, threads, parity_record, parity_gate, scratch_parent=None):
    cpu_setup(threads)
    import data
    import native_protocol as protocol
    from transformers import AutoTokenizer
    from repeats import repeats
    from run import export_cache, pack_context
    from cache import paths, compare_cache
    scratch = Path(tempfile.mkdtemp(prefix=f'dna-r{rank}-c{client}-', dir=scratch_parent))
    active = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        sampler = sampler_from_source(model_path)
        vocab = (Path(model_path) / 'vocab.txt').read_text().splitlines()
        proxy = ForwardProxy(client, requests, replies, fatal)
        def evaluate(record):
            record = dict(record)
            record['repeat_tokens'] = [i+1 for i, yes in enumerate(repeats(record['sequence'])) if yes]
            proxy.reset(record['index'])
            protocol.run_context(record, proxy, tokenizer, sampler, vocab, scratch, 32)
            directory = scratch / f'e{record["index"]:04d}'
            result, payload = export_cache(directory, proxy.hidden, record)
            assert proxy.calls == result['summary']['forward_batches']
            assert proxy.sequences == result['summary']['sequence_forwards']
            return directory, result, payload
        if parity_record is not None:
            active = parity_record['index']
            directory, result, payload = evaluate(parity_record)
            _, archive = paths(state, active)
            check = compare_cache(directory, archive)
            data.atomic(run_dir / f'parity-rank{rank}.json', dict(index=active, status='PASS', **check))
            events.put(dict(event='parity_pass', rank=rank, index=active, **check))
            # Old archive is untouched; newly computed parity copy is kept for audit.
        while not parity_gate.wait(.5):
            if fatal.is_set() or stop.is_set():
                return
        while not stop.is_set() and not fatal.is_set():
            try:
                record = tasks.get(timeout=.5)
            except queue.Empty:
                continue
            if record is None:
                break
            active = record['index']
            result_path, archive = paths(state, active)
            assert not archive.exists() and not result_path.exists(), f'Duplicate context {active}'
            if shutil.disk_usage(state).free < 251 * 2**30:
                events.put(dict(event='disk_reserve_stop', index=active)); stop.set(); break
            started = time.monotonic()
            events.put(dict(event='context_started', index=active, rank=rank, client=client))
            directory, result, payload = evaluate(record)
            archive.parent.mkdir(parents=True, exist_ok=True)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            digest = pack_context(directory, archive)
            result['archive_sha256'] = digest
            data.atomic(result_path, result)
            events.put(dict(event='context_complete', index=active, rank=rank, client=client,
                            archive_sha256=digest, physical_calls=proxy.calls,
                            sequence_forwards=proxy.sequences, seconds=time.monotonic()-started))
            # Delete only this freshly generated, durably archived context directory.
            assert directory.parent == scratch and directory.name == f'e{active:04d}'
            shutil.rmtree(directory)
    except BaseException:
        events.put(dict(event='error', role='cpu', rank=rank, client=client, index=active,
                        scratch=str(scratch), traceback=traceback.format_exc()))
        fatal.set()
        raise
    finally:
        events.put(dict(event='client_stopped', rank=rank, client=client, scratch=str(scratch)))
