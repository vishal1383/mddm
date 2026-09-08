"""CPU-only reports from saved caches; never calls the DNA model."""
import argparse
import fcntl
import io
import json
import os
import sys
import tarfile
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dna_anchor_census'))
import numpy as np
import torch
from report import summarize, example_plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    lock = (args.state / 'plots.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    summary = summarize(args.state)
    for marker in sorted((args.state / 'results').glob('part*/e*.json')):
        index = int(marker.stem[1:])
        if index >= 5 and index % 1000 != 0:
            continue
        result = json.loads(marker.read_text())
        archive = args.state / 'cache' / marker.parent.name / (marker.stem + '.tar')
        with tarfile.open(archive) as tar:
            with tar.extractfile('probability_changes.npz') as f:
                with np.load(io.BytesIO(f.read())) as arrays:
                    payload = {k: arrays[k] for k in arrays.files}
        example_plots(result, payload, args.state / 'plots' / marker.stem)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
