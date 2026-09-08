"""D3LM single-anchor interventions: exhaustive positions, IG and count teachers.

Adapted from the local 500-sequence D3LM audit; no gold anchor injection.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

H = Path(__file__).resolve().parent
A = H.parent/'d3lm_pretrained_probe_001/assets/from_nt'
THRESHOLD = .95
FULL_PROB_AUDIT_STRIDE = 500
FULL_PROB_POSITIVE_LIMIT = 2


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')


def stats(p, gold):
    conf, ids = p.max(-1)
    return dict(confidence=conf, prediction=ids,
                entropy=-(p.double()*p.double().clamp_min(1e-30).log2()).sum(-1),
                reference_probability=p.gather(1, gold[:, None]).squeeze(1),
                reference_correct=ids.eq(gold),
                max_norm_error=float((p.sum(-1)-1).abs().max()))


def metrics(root, child, indices):
    ix = torch.tensor(indices, dtype=torch.long)
    new = (child['confidence'][ix] >= THRESHOLD) & (root['confidence'][ix] < THRESHOLD)
    lost = (child['confidence'][ix] < THRESHOLD) & (root['confidence'][ix] >= THRESHOLD)
    before_correct = (root['confidence'][ix] >= THRESHOLD) & root['reference_correct'][ix]
    after_correct = (child['confidence'][ix] >= THRESHOLD) & child['reference_correct'][ix]
    before_wrong = (root['confidence'][ix] >= THRESHOLD) & ~root['reference_correct'][ix]
    after_wrong = (child['confidence'][ix] >= THRESHOLD) & ~child['reference_correct'][ix]
    n, l = int(new.sum()), int(lost.sum())
    return dict(new_tokens=n, lost_tokens=l, net_tokens=n-l,
                correct_before=int(before_correct.sum()), correct_after=int(after_correct.sum()),
                new_correct=int((after_correct & ~before_correct).sum()),
                new_wrong=int((after_wrong & ~before_wrong).sum()),
                new_confident_correct=int((new & child['reference_correct'][ix]).sum()),
                new_confident_wrong=int((new & ~child['reference_correct'][ix]).sum()),
                max_probability_gain=float((child['confidence'][ix]-root['confidence'][ix]).sum()),
                reference_probability_gain=float((child['reference_probability'][ix]-root['reference_probability'][ix]).sum()),
                entropy_drop=float((root['entropy'][ix]-child['entropy'][ix]).sum()),
                new_indices=ix[new].tolist())


def choose_candidates(masked, p, ent, conf, seed):
    eligible = [i for i in masked if float(conf[i]) <= .5]
    fallback = not eligible
    if fallback:
        eligible = list(masked)
    rg = random.Random(seed)
    chosen, reasons = [], {}
    def add(items, reason, limit):
        added = 0
        for i in items:
            if i not in chosen:
                chosen.append(i); reasons[str(i)] = reason; added += 1
                if added == limit:
                    break
    add(sorted(eligible, key=lambda i: (-float(ent[i]), i)), 'entropy', 2)
    mset = set(masked)
    overlap = {}
    for i in eligible:
        neighbors = [j for j in range(max(1, i-3), i+4) if j in mset and j != i]
        overlap[i] = max((float((p[i]*p[j]).sum()) for j in neighbors), default=0.)
    add(sorted(eligible, key=lambda i: (-overlap[i], i)), 'nearby_distribution_overlap', 2)
    add([eligible[round((len(eligible)-1)*q)] for q in (.25, .75)], 'spatial', 2)
    add(rg.sample(eligible, len(eligible)), 'random_fill', max(0, 8-len(chosen)))
    return chosen[:8], reasons, fallback


def run_context(ex, model, tokenizer, sampler, vocab, out, batch_size):
    started = time.time()
    d = out/f'e{ex["index"]:04d}'
    if (d/'result.json').exists():
        return json.loads((d/'result.json').read_text())['summary']
    d.mkdir(exist_ok=False)
    gold = torch.tensor(tokenizer(ex['sequence'])['input_ids'])
    words = tokenizer.convert_ids_to_tokens(gold.tolist())
    assert ''.join(words[1:]) == ex['sequence']
    full_positions = [i for i, w in enumerate(words) if len(w)==6 and set(w)<=set('ACGT')]
    assert full_positions == list(range(1, 342))
    masked = sorted(random.Random(ex['mask_seed']).sample(full_positions, 170))
    x = gold.clone(); x[masked] = tokenizer.mask_token_id
    torch.save(x, d/'root_input.pt')
    torch.save(gold, d/'reference_ids.pt')
    with torch.inference_mode():
        logits = model(input_ids=x[None], attention_mask=torch.ones_like(x[None])).logits[0].float()
    assert torch.isfinite(logits).all()
    p0 = logits.softmax(-1)
    root = stats(p0[masked], gold[masked])
    assert root['max_norm_error'] < 1e-5
    torch.save(logits[masked], d/'root_logits.pt')
    torch.save(root, d/'root_stats.pt')
    ent = -(p0*p0.clamp_min(1e-30).log2()).sum(-1)
    conf = p0.max(-1).values
    candidates = list(masked)
    reasons = {str(pos):'all_masked_positions' for pos in candidates}
    fallback = False
    baselines = dict(confidence=max(masked, key=lambda i: (float(conf[i]), -i)),
                     random=random.Random(ex['random_seed']).choice(masked), earliest=masked[0])
    root_meta = dict(index=ex['index'], row_index=ex['row_index'], sha256=ex['sha256'],
                     species=ex['species'], masked=masked, candidates=candidates,
                     candidate_reasons=reasons, uncertain_fallback=fallback, baselines=baselines,
                     repeat_tokens=ex['repeat_tokens'], threshold=THRESHOLD,
                     root_confident_tokens=int((root['confidence']>=THRESHOLD).sum()),
                     discovery_seeds=ex['discovery_seeds'], evaluation_seeds=ex['evaluation_seeds'])
    dump(d/'context.json', root_meta)
    cache, unique_counts, batch_counts = {}, dict(discovery=0, evaluation=0), dict(discovery=0, evaluation=0)
    pos_to_ix = {pos: j for j, pos in enumerate(masked)}
    def proposals(seeds, phase):
        values = {}
        for seed in seeds:
            torch.manual_seed(seed)
            _, ids = sampler(logits[masked], temperature=1., top_p=1., top_k=0)
            full = x.clone(); full[masked] = ids
            values[seed] = full
        torch.save(values, d/f'{phase}_proposals.pt')
        return values
    positive_full_saved = 0
    def children(pairs, phase):
        nonlocal positive_full_saved
        missing = list(dict.fromkeys(pair for pair in pairs if pair not in cache))
        for lo in range(0, len(missing), batch_size):
            items = missing[lo:lo+batch_size]
            ys = x[None].repeat(len(items), 1)
            for k, (pos, value) in enumerate(items):
                ys[k, pos] = value
            with torch.inference_mode():
                raw = model(input_ids=ys, attention_mask=torch.ones_like(ys)).logits.float()
            assert torch.isfinite(raw).all()
            probs = raw[:, masked].softmax(-1)
            for k, pair in enumerate(items):
                pos, value = pair
                c = stats(probs[k], gold[masked])
                assert c['max_norm_error'] < 1e-5
                cache[pair] = c
                key = f'p{pos}_v{value}'
                torch.save(dict(input_ids=ys[k].clone(), statistics=c, phase_first_evaluated=phase), d/f'{key}.pt')
                probe_indices = [j for j, m in enumerate(masked) if m != pos]
                audit_full = ex['index']%FULL_PROB_AUDIT_STRIDE == 0
                positive_full = (positive_full_saved < FULL_PROB_POSITIVE_LIMIT and
                                 metrics(root, c, probe_indices)['new_tokens'] > 0)
                if audit_full or positive_full:
                    torch.save(probs[k].clone(), d/f'{key}_full_probs.pt')
                    positive_full_saved += int(not audit_full)
            unique_counts[phase] += len(items); batch_counts[phase] += 1
    disc = proposals(ex['discovery_seeds'], 'discovery')
    children([(pos, int(values[pos])) for values in disc.values() for pos in candidates], 'discovery')
    discovery = []
    candidate_scores = {}
    for pos in candidates:
        rr = []
        for seed, proposal in disc.items():
            value = int(proposal[pos])
            score = metrics(root, cache[(pos,value)], [j for j,m in enumerate(masked) if m != pos])
            rr.append(score)
            discovery.append(dict(position=pos, seed=seed, value=value, **score))
        candidate_scores[pos] = tuple(sum(r[k] for r in rr)/len(rr) for k in ('new_tokens','net_tokens','entropy_drop'))
    anchor = max(candidates, key=lambda pos: (*candidate_scores[pos], -pos))
    ig_anchor = max(candidates, key=lambda pos: (candidate_scores[pos][2], candidate_scores[pos][0], -pos))
    actions = dict(max_new=anchor, ig=ig_anchor, **baselines)
    common = [pos for pos in masked if pos not in set(actions.values())]
    common_indices = [pos_to_ix[pos] for pos in common]
    # Persist the decision before drawing fresh evaluation values.
    selection = dict(actions=actions, common=common, candidate_scores=candidate_scores,
                     tie_break_max_new=['mean_new_tokens','mean_net_tokens','mean_IG_bits','lowest_position'],
                     tie_break_ig=['mean_IG_bits','mean_new_tokens','lowest_position'],
                     anchor_root_maxprob=float(conf[anchor]), anchor_root_entropy=float(ent[anchor]),
                     discovery=discovery)
    dump(d/'selection_before_evaluation.json', selection)
    ev = proposals(ex['evaluation_seeds'], 'evaluation')
    children([(pos, int(values[pos])) for values in ev.values() for pos in actions.values()], 'evaluation')
    rows = []
    repeat_set = set(ex['repeat_tokens'])
    for seed, proposal in ev.items():
        for arm, pos in actions.items():
            value = int(proposal[pos]); child = cache[(pos,value)]
            met = metrics(root, child, common_indices)
            unlocked = []
            for ix in met['new_indices']:
                target = masked[ix]
                prediction = int(child['prediction'][ix])
                unlocked.append(dict(position=target, token=vocab[prediction],
                                     before_pmax=float(root['confidence'][ix]),
                                     before_prediction_probability=float(p0[target,prediction]),
                                     after_pmax=float(child['confidence'][ix]),
                                     reference_match=prediction == int(gold[target]),
                                     repeat=target in repeat_set, distance=abs(target-pos)))
            rows.append(dict(arm=arm, seed=seed, position=pos, sampled_id=value, token=vocab[value],
                             sampled_probability=float(p0[pos,value]), root_maxprob=float(conf[pos]),
                             root_entropy=float(ent[pos]), anchor_repeat=pos in repeat_set,
                             full_sixmer=len(vocab[value])==6 and set(vocab[value])<=set('ACGT'),
                             actually_unmasked=value != tokenizer.mask_token_id,
                             **met, unlocked=unlocked))
    summary = dict(index=ex['index'], species=ex['species'], row_index=ex['row_index'],
                   sequence_has_repeat=bool(ex['repeat_tokens']), common_count=len(common),
                   logical_child_evaluations=len(discovery)+len(rows), unique_children=unique_counts,
                   forward_batches=1+sum(batch_counts.values()),
                   sequence_forwards=1+sum(unique_counts.values()),
                   elapsed_seconds=time.time()-started,
                   arms={arm:dict(mean_new_tokens=sum(r['new_tokens'] for r in rows if r['arm']==arm)/8,
                                  samples_with_unlock=sum(r['new_tokens']>0 for r in rows if r['arm']==arm),
                                  position=pos) for arm,pos in actions.items()})
    dump(d/'result.json', dict(context=root_meta, selection=selection, rows=rows, summary=summary))
    return summary
