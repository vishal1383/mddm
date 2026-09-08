"""Single-A100 DNA anchor analysis and reusable teacher-cache generation."""
import argparse
import fcntl
import importlib.metadata
import json
import os
import shutil
import signal
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForMaskedLM,AutoTokenizer

import data
import native_protocol as protocol
from repeats import repeats

H = Path(__file__).resolve().parent
ARMS = ['max_new','ig','confidence','random','earliest']


def require_a100(torch_module=torch):
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count()!=1:
        raise RuntimeError('This job requires exactly one visible CUDA GPU.')
    name = torch_module.cuda.get_device_name(0)
    if 'A100' not in name.upper():
        raise RuntimeError(f'Refusing non-A100 allocation: {name}')
    return name


def export_cache(directory,hidden,record):
    result = json.loads((directory/'result.json').read_text())
    context,selection = result['context'],result['selection']
    masked,common = context['masked'],selection['common']
    root = torch.load(directory/'root_stats.pt',weights_only=True)
    logits = torch.load(directory/'root_logits.pt',weights_only=True)
    gold = torch.load(directory/'reference_ids.pt',weights_only=True)
    root_p = logits.softmax(-1)
    lookup = {pos:i for i,pos in enumerate(masked)}
    ix = torch.tensor([lookup[pos] for pos in common])
    torch.save(dict(last_hidden_state=hidden[masked].float(),masked_positions=torch.tensor(masked),
                    feature_scope='root before any anchor; frozen-Base hidden states',
                    final_validity_target=None,remaining_NFE_target=None),directory/'root_features.pt')
    keys = ['before_same_token_probability','after_same_token_probability','before_max_probability',
            'after_max_probability','before_reference_probability','after_reference_probability',
            'before_entropy_bits','after_entropy_bits','prediction_ids']
    arrays = {k:[] for k in keys}
    for row in result['rows']:
        st = torch.load(directory/f'p{row["position"]}_v{row["sampled_id"]}.pt',weights_only=True)['statistics']
        pred = st['prediction'][ix]
        values = [root_p[ix,pred],st['confidence'][ix],root['confidence'][ix],st['confidence'][ix],
                  root['reference_probability'][ix],st['reference_probability'][ix],
                  root['entropy'][ix],st['entropy'][ix],pred]
        for key,value in zip(keys,values): arrays[key].append(value.numpy())
        row['new_tokens_090'] = int(((root['confidence'][ix]<.9)&(st['confidence'][ix]>=.9)).sum())
        row['reference_correct_new_095'] = row['new_confident_correct']
    payload = {k:np.stack(v) for k,v in arrays.items()}
    payload.update(common_positions=np.asarray(common),arm=np.asarray([r['arm'] for r in result['rows']]),
                   seed=np.asarray([r['seed'] for r in result['rows']],dtype=np.int64))
    np.savez_compressed(directory/'probability_changes.npz',**payload)
    # Keep the old reference-correct GSM8K key as a distinctly labeled target.
    # No reference nucleotide is inserted by this diagnostic or by inference.
    legacy = {}
    candidates = selection['candidate_scores']
    for pos_string in candidates:
        pos = int(pos_string)
        rows = [r for r in selection['discovery'] if r['position']==pos]
        after = sum(r['correct_after'] for r in rows)/len(rows)
        gain = sum(r['correct_after']-r['correct_before'] for r in rows)/len(rows)
        wrong = sum(r['new_wrong'] for r in rows)/len(rows)
        lp = float(root_p[lookup[pos],int(gold[pos])].clamp_min(1e-30).log())
        legacy[pos] = (gain,gain,after,-wrong,lp,-pos)
    targets = dict(schema='dna_singleton_two_teacher_cache_v1',index=record['index'],
        sequence_sha256=record['sha256'],split=record['split'],model=data.MODEL,model_revision=data.MODEL_REV,
        main_threshold=.95,secondary_report_threshold=.90,
        candidate_positions=context['candidates'],candidate_count=len(context['candidates']),
        candidate_coverage='all 170 masked positions; two ordinary sampled values per position',
        source_reference_used_as_anchor=False,selection_uses_fresh_evaluation_outcomes=False,
        chosen_ig_position=selection['actions']['ig'],chosen_max_new_position=selection['actions']['max_new'],
        max_new_tie_break=selection['tie_break_max_new'],ig_tie_break=selection['tie_break_ig'],
        discovery_candidate_scores=candidates,all_discovery_outcomes=selection['discovery'],
        reference_correct_legacy_gsm8k_key={str(k):list(v) for k,v in legacy.items()},
        reference_correct_legacy_gsm8k_chosen_position=max(legacy,key=lambda pos:legacy[pos]),
        reference_key_note='Oracle/default gain ranking formula from precompute_threshold_unlock_targets.candidate_key, applied to ordinary-proposal interventions, not gold-anchor interventions; source agreement is not biological validity.',
        observed_fresh_rows=result['rows'],reference_ids_file='reference_ids.pt',
        root_features_file='root_features.pt',root_logits_file='root_logits.pt',
        native_discovery_proposals_file='discovery_proposals.pt',native_evaluation_proposals_file='evaluation_proposals.pt',
        all_sampled_singleton_children='p{position}_v{sampled_token_id}.pt',
        same_root_complete_pair_action_targets_available=False,completed_generation_returns_available=False)
    data.atomic(directory/'training_targets.json',targets)
    result['dataset'] = {k:v for k,v in record.items() if k not in ['sequence','repeat_tokens']}
    data.atomic(directory/'result.json',result)
    return result,payload


def pack_context(directory,archive):
    assert not archive.exists()
    artifacts = {p.name:data.sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
    data.atomic(directory/'artifact_manifest.json',artifacts)
    temp = archive.with_suffix('.tar.partial')
    if temp.exists():
        temp.rename(temp.with_name(temp.name+f'.preserved.{time.time_ns()}'))
    with tarfile.open(temp,'w') as tar:
        for path in sorted(directory.iterdir()): tar.add(path,arcname=path.name,recursive=False)
    with tarfile.open(temp,'r') as tar:
        for name,digest in artifacts.items():
            with tar.extractfile(name) as f:
                import hashlib
                assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    temp.replace(archive)
    return data.sha(archive)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--scope',choices=['all_species','four_species'],default='all_species')
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--limit',type=int,help='Explicit limited diagnostic only; production omits this.')
    ap.add_argument('--report-every',type=int,default=1000)
    args = ap.parse_args()
    gpu_name = require_a100()
    state = args.state.resolve(); state.mkdir(parents=True,exist_ok=True)
    lock = (state/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model_path,source = data.fetch(state/'assets')
    records,inventory = data.records(source,args.scope)
    if args.limit is not None: records = records[:args.limit]
    source_sha = {p.name:data.sha(p) for p in H.iterdir() if p.suffix in ('.py','.slurm','.txt')}
    contract = dict(model=data.MODEL,model_revision=data.MODEL_REV,model_sha256=data.MODEL_SHA,
                    dataset=data.DATASET,dataset_revision=data.DATA_REV,dataset_sha256=data.DATA_SHA,
                    scope=args.scope,expected_contexts=len(records),full_scope_contexts=inventory['eligible_unique'],
                    source_sha256=source_sha,mask_positions=170,sequence_bases=2048,
                    tokenization='complete native sequence; no truncation',threshold=.95,
                    temperature=1.,top_p=1.,top_k=0,batch_size=args.batch_size,
                    discovery_draws_per_position=2,fresh_draws_per_arm=8,candidate_positions='all masked',
                    arms=ARMS,training_steps=0,precision='float32; TF32 disabled',device_name=gpu_name,
                    versions={k:importlib.metadata.version(k) for k in ['torch','transformers','tokenizers','huggingface-hub']})
    if (state/'contract.json').exists():
        assert json.loads((state/'contract.json').read_text())==contract,'Do not mix code/model/protocol revisions in one cache.'
    else:
        data.atomic(state/'contract.json',contract); data.atomic(state/'inventory.json',inventory)
        with (state/'dataset_manifest.jsonl').open('x') as f:
            for r in records: f.write(json.dumps({k:v for k,v in r.items() if k!='sequence'})+'\n')
    model,loading = AutoModelForMaskedLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=True,
                      torch_dtype=torch.float32,output_loading_info=True)
    assert not any(loading[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model = model.eval().requires_grad_(False).to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    sampler = sys.modules[type(model).__module__]._sample_tokens
    vocab = (model_path/'vocab.txt').read_text().splitlines()
    stop = False
    def stop_after_context(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGUSR1,stop_after_context); signal.signal(signal.SIGTERM,stop_after_context)
    invocation = str(time.time_ns()); journal = (state/f'forward_events_{invocation}.jsonl').open('x',buffering=1)
    def event(kind,**kwargs):
        journal.write(json.dumps(dict(event=kind,unix=time.time(),**kwargs))+'\n'); journal.flush(); os.fsync(journal.fileno())
    scratch = Path(tempfile.mkdtemp(prefix='d3lm-anchor-',dir=os.environ.get('TMPDIR','/tmp')))
    completed = calls = sequences = 0; started=time.time(); active=None; hidden=None; context_calls=0
    class GPUForward:
        def __call__(self,**kwargs):
            nonlocal calls,sequences,hidden,context_calls
            root = context_calls==0; context_calls+=1; calls+=1
            n=kwargs['input_ids'].shape[0]; sequences+=n
            event('forward_started',index=active,call=calls,sequences=n)
            output=model(**{k:v.to('cuda:0') for k,v in kwargs.items()},output_hidden_states=root)
            logits=output.logits.float().cpu()
            if root: hidden=output.hidden_states[-1][0].float().cpu()
            event('forward_finished',index=active,call=calls,sequences=n)
            return SimpleNamespace(logits=logits)
    data.atomic(state/'status.json',dict(status='RUNNING',gpu=gpu_name,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                                       expected=len(records),completed=0,scope=args.scope))
    try:
        for record in records:
            active=record['index']; part=f'part{active//1000:04d}'; name=f'e{active:06d}'
            result_dir=state/'results'/part; archive_dir=state/'cache'/part
            result_dir.mkdir(parents=True,exist_ok=True); archive_dir.mkdir(parents=True,exist_ok=True)
            result_path=result_dir/f'{name}.json'; archive=archive_dir/f'{name}.tar'
            if result_path.exists():
                r=json.loads(result_path.read_text())
                assert r['context']['sha256']==record['sha256'] and archive.exists()
                completed+=1; continue
            if stop: break
            if shutil.disk_usage(state).free < 251*2**30:
                event('disk_reserve_stop',completed=completed); stop=True; break
            # A durable archive may precede its completion marker after interruption.
            if archive.exists():
                with tarfile.open(archive) as tar:
                    r=json.load(tar.extractfile('result.json'))
                assert r['context']['sha256']==record['sha256']
                r['archive_sha256']=data.sha(archive); data.atomic(result_path,r); completed+=1; continue
            record['repeat_tokens']=[i+1 for i,yes in enumerate(repeats(record['sequence'])) if yes]
            context_calls=0; before_calls=calls; before_sequences=sequences
            protocol.run_context(record,GPUForward(),tokenizer,sampler,vocab,scratch,args.batch_size)
            directory=scratch/f'e{active:04d}'
            r,payload=export_cache(directory,hidden,record)
            assert calls-before_calls==r['summary']['forward_batches']
            assert sequences-before_sequences==r['summary']['sequence_forwards']
            if active<5 or active%1000==0:
                from report import example_plots
                example_plots(r,payload,state/'plots'/name)
            digest=pack_context(directory,archive)
            r['archive_sha256']=digest; data.atomic(result_path,r)
            event('context_complete',index=active,archive_sha256=digest,physical_calls=calls-before_calls,
                  physical_sequence_forwards=sequences-before_sequences)
            # Only generated scratch files whose complete archive was verified.
            assert directory.parent==scratch and directory.name==f'e{active:04d}'
            shutil.rmtree(directory)
            completed+=1
            if completed%25==0 or completed==1:
                status=dict(status='RUNNING',gpu=gpu_name,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                            completed=completed,expected=len(records),physical_calls_this_invocation=calls,
                            sequence_forwards_this_invocation=sequences,elapsed_seconds=time.time()-started)
                data.atomic(state/'status.json',status); print(json.dumps(status),flush=True)
            if completed%args.report_every==0:
                from report import summarize
                summarize(state)
        from report import summarize
        summarize(state)
        status='COMPLETE_LIMITED_DIAGNOSTIC' if args.limit and completed==len(records) else (
               'COMPLETE_CENSUS' if completed==len(records) else 'INTERRUPTED_RESUMABLE')
        data.atomic(state/'status.json',dict(status=status,completed=completed,expected=len(records),gpu=gpu_name,
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'),physical_calls_this_invocation=calls,
                    sequence_forwards_this_invocation=sequences,new_model_training_steps=0))
        print(json.dumps(dict(status=status,completed=completed)),flush=True)
    except BaseException as exc:
        data.atomic(state/'status.json',dict(status='FAILED_RECOVERABLE',index=active,completed=completed,
                    error=repr(exc),physical_calls_this_invocation=calls,sequence_forwards_this_invocation=sequences,
                    scratch=str(scratch),slurm_job_id=os.environ.get('SLURM_JOB_ID')))
        raise
    finally:
        journal.close(); lock.close()


if __name__=='__main__': main()
