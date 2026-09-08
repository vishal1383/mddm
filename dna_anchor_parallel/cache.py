"""Compatibility and completion validation for the unchanged singleton cache."""
import importlib.metadata
import json
import sys
import tarfile
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[1] / 'dna_anchor_census'
sys.path.insert(0, str(LEGACY))
import data


def validate_contract(state, records):
    contract = json.loads((state / 'contract.json').read_text())
    sources = {p.name: data.sha(p) for p in LEGACY.iterdir() if p.suffix in ('.py', '.slurm', '.txt')}
    assert sources == contract['source_sha256'], 'Legacy experiment source changed'
    expected = dict(model=data.MODEL, model_revision=data.MODEL_REV, model_sha256=data.MODEL_SHA,
                    dataset=data.DATASET, dataset_revision=data.DATA_REV, dataset_sha256=data.DATA_SHA,
                    scope='all_species', expected_contexts=len(records), full_scope_contexts=len(records),
                    mask_positions=170, sequence_bases=2048, batch_size=32, threshold=.95,
                    discovery_draws_per_position=2, fresh_draws_per_arm=8, training_steps=0,
                    precision='float32; TF32 disabled', temperature=1., top_p=1., top_k=0,
                    candidate_positions='all masked', arms=['max_new', 'ig', 'confidence', 'random', 'earliest'],
                    tokenization='complete native sequence; no truncation')
    for key, value in expected.items():
        assert contract[key] == value, (key, contract[key], value)
    for package, version in contract['versions'].items():
        assert importlib.metadata.version(package) == version, (package, version)
    assert data.sha(state / 'assets/model/model.safetensors') == data.MODEL_SHA
    with (state / 'dataset_manifest.jsonl').open() as f:
        count = 0
        for count, line in enumerate(f, 1):
            assert json.loads(line) == {k: v for k, v in records[count-1].items() if k != 'sequence'}
    assert count == len(records)
    return contract


def paths(state, index):
    part, name = f'part{index // 1000:04d}', f'e{index:06d}'
    return state / 'results' / part / (name + '.json'), state / 'cache' / part / (name + '.tar')


def completed_indices(state, records):
    """Validate ownership and recover archives saved just before interruption."""
    complete = set()
    for archive in sorted((state / 'cache').glob('part*/e*.tar')):
        index = int(archive.stem[1:])
        result_path, expected_archive = paths(state, index)
        assert archive == expected_archive and 0 <= index < len(records)
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            with tarfile.open(archive) as tar:
                result = json.load(tar.extractfile('result.json'))
            result['archive_sha256'] = data.sha(archive)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            data.atomic(result_path, result)
        assert result['context']['index'] == index
        assert result['context']['sha256'] == records[index]['sha256']
        assert len(result['archive_sha256']) == 64
        complete.add(index)
    for marker in (state / 'results').glob('part*/e*.json'):
        assert int(marker.stem[1:]) in complete, f'Missing archive for {marker}'
    return complete


def compare_cache(directory, archive, atol=2e-5):
    """Strict discrete parity, tight numerical parity; timing/serialization may differ."""
    import io
    import numpy as np
    import torch
    checked = dict(tensors=0, numbers=0, max_abs_error=0.)
    def compare(a, b, trail=''):
        if isinstance(a, torch.Tensor):
            assert a.shape == b.shape and a.dtype == b.dtype, trail
            if a.dtype.is_floating_point:
                delta = float((a-b).abs().max()) if a.numel() else 0.
                checked['max_abs_error'] = max(checked['max_abs_error'], delta)
                assert torch.allclose(a, b, atol=atol, rtol=1e-6), (trail, delta)
            else:
                assert torch.equal(a, b), trail
            checked['tensors'] += 1
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), trail
            for k in a:
                if k not in {'elapsed_seconds', 'archive_sha256'}:
                    compare(a[k], b[k], f'{trail}/{k}')
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), trail
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f'{trail}/{i}')
        elif isinstance(a, float):
            delta = abs(a-b)
            checked['max_abs_error'] = max(checked['max_abs_error'], delta)
            assert delta <= atol + abs(a)*1e-6, (trail, a, b)
            checked['numbers'] += 1
        else:
            assert a == b, (trail, a, b)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = member.name
            if name.endswith('.pt'):
                with tar.extractfile(member) as f:
                    old = torch.load(f, weights_only=True)
                compare(old, torch.load(directory / name, weights_only=True), name)
            elif name.endswith('.json') and name != 'artifact_manifest.json':
                with tar.extractfile(member) as f:
                    old = json.load(f)
                compare(old, json.loads((directory / name).read_text()), name)
            elif name.endswith('.npz'):
                with tar.extractfile(member) as f:
                    with np.load(io.BytesIO(f.read())) as old, np.load(directory / name) as new:
                        assert old.files == new.files, name
                        for key in old.files:
                            a, b = old[key], new[key]
                            if np.issubdtype(a.dtype, np.number):
                                compare(torch.from_numpy(a), torch.from_numpy(b), name+'/'+key)
                            else:
                                assert np.array_equal(a, b), name+'/'+key
    return checked
