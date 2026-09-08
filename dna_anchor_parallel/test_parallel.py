import json
import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ForwardProxy, serve
from cache import paths, completed_indices, compare_cache


class ParallelTest(unittest.TestCase):
    def test_gpu_request_roundtrip_and_root_hidden(self):
        ctx = mp.get_context('spawn')
        fatal, shutdown = ctx.Event(), ctx.Event()
        requests, events = ctx.Queue(2), ctx.Queue()
        replies = [ctx.Queue(1), ctx.Queue(1)]
        with tempfile.TemporaryDirectory(prefix='dna-ipc-test-') as tmp:
            server = ctx.Process(target=serve, args=(0, '', None, requests, replies, events,
                fatal, shutdown, Path(tmp)/'journal.jsonl', True))
            server.start()
            try:
                ready = events.get(timeout=30)
                self.assertEqual(ready['event'], 'gpu_ready')
                for client in [0, 1, 0, 1]:
                    model = ForwardProxy(client, requests, replies[client], fatal, timeout=15)
                    model.reset(10+client)
                    x = torch.tensor([[1, 2, 3]]) + client
                    output = model(input_ids=x, attention_mask=torch.ones_like(x))
                    self.assertTrue(torch.equal(output.logits[:, :, 0], x.float()))
                    self.assertTrue(torch.equal(model.hidden[:, 0], x[0].float()))
                    model(input_ids=x.repeat(3, 1), attention_mask=x.repeat(3, 1))
                    self.assertEqual(model.calls, 2)
                    self.assertEqual(model.sequences, 4)
                del output, model
            finally:
                shutdown.set(); server.join(timeout=60)
                if server.is_alive():
                    server.terminate(); server.join()
            self.assertEqual(server.exitcode, 0)

    def test_gpu_error_fails_fast(self):
        ctx = mp.get_context('spawn')
        fatal = ctx.Event(); fatal.set()
        proxy = ForwardProxy(0, ctx.Queue(1), ctx.Queue(1), fatal, timeout=1)
        with self.assertRaisesRegex(RuntimeError, 'pipeline failed'):
            proxy(input_ids=torch.tensor([[1]]))

    def test_disjoint_paths_across_partition_boundary(self):
        self.assertEqual(paths(Path('/cache'), 1000)[0], Path('/cache/results/part0001/e001000.json'))
        self.assertNotEqual(paths(Path('/cache'), 999), paths(Path('/cache'), 1000))

    def test_resume_recovery_and_strict_cache_parity(self):
        import tarfile
        from run import pack_context
        with tempfile.TemporaryDirectory(prefix='dna-resume-test-') as tmp:
            state = Path(tmp); context = state/'scratch'; context.mkdir()
            result = dict(context=dict(index=0, sha256='abc'), summary=dict(elapsed_seconds=1.))
            (context/'result.json').write_text(json.dumps(result))
            torch.save(torch.tensor([2, 4]), context/'proposals.pt')
            marker, archive = paths(state, 0)
            archive.parent.mkdir(parents=True)
            pack_context(context, archive)
            self.assertEqual(completed_indices(state, [dict(sha256='abc')]), {0})
            self.assertTrue(marker.exists())
            self.assertEqual(compare_cache(context, archive)['tensors'], 1)
            torch.save(torch.tensor([2, 5]), context/'proposals.pt')
            with self.assertRaises(AssertionError):
                compare_cache(context, archive)
            with self.assertRaises(AssertionError):
                completed_indices(state, [dict(sha256='wrong')])

    def test_gpu_job_has_no_plot_calls_and_exact_gpu_resources(self):
        here = Path(__file__).resolve().parent
        for name in ['run.py', 'worker.py']:
            source = (here/name).read_text()
            self.assertNotIn('from report import', source)
            self.assertNotIn('example_plots(', source)
            self.assertNotIn('summarize(', source)
        source = (here/'run_unity.sbatch').read_text()
        for directive in ['--gres=gpu:a100:4', '--constraint=a100-80g', '--partition=gpu-preempt', '--time=45:00:00']:
            self.assertIn('#SBATCH '+directive, source)
        self.assertNotIn('#SBATCH --gres=', (here/'plots_unity.sbatch').read_text())


if __name__ == '__main__':
    unittest.main()
