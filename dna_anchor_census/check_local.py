"""One real-record CPU correctness check, not a throughput benchmark."""
import argparse
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForMaskedLM,AutoTokenizer
import data
import native_protocol as protocol
from run import export_cache,pack_context
from report import example_plots
from repeats import repeats


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True); ap.add_argument('--old-manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    args.out.mkdir(exist_ok=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    records,inventory=data.records(args.source)
    prior=json.loads(args.old_manifest.read_text())['examples']
    for r,old in zip(records[:500],prior):
        for key in ['index','row_index','sha256','species','sequence','mask_seed','discovery_seeds','evaluation_seeds']:
            assert r[key]==old[key],key
    assert sum(r['in_original_four_species'] for r in records)==61236
    assert sum(r['development'] for r in records)==505
    model,loading=AutoModelForMaskedLM.from_pretrained(args.assets,local_files_only=True,trust_remote_code=True,
                    torch_dtype=torch.float32,output_loading_info=True)
    assert not any(loading[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model.eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(args.assets,local_files_only=True)
    sampler=sys.modules[type(model).__module__]._sample_tokens
    vocab=(args.assets/'vocab.txt').read_text().splitlines()
    r=records[0]; r['repeat_tokens']=[i+1 for i,yes in enumerate(repeats(r['sequence'])) if yes]
    calls=seqs=0; hidden=None
    class CPU:
        def __call__(self,**kwargs):
            nonlocal calls,seqs,hidden
            root=calls==0; calls+=1; seqs+=kwargs['input_ids'].shape[0]
            out=model(**kwargs,output_hidden_states=root)
            if root: hidden=out.hidden_states[-1][0].detach().cpu()
            return out
    protocol.run_context(r,CPU(),tokenizer,sampler,vocab,args.out,32)
    result,changes=export_cache(args.out/'e0000',hidden,r)
    assert result['summary']['forward_batches']==calls and result['summary']['sequence_forwards']==seqs
    assert len(result['context']['candidates'])==170 and len(result['rows'])==40
    assert len(result['selection']['discovery'])==340
    assert changes['after_max_probability'].shape[0]==40
    assert not set(result['selection']['common']) & set(result['selection']['actions'].values())
    assert (args.out/'e0000/selection_before_evaluation.json').stat().st_mtime_ns <= (args.out/'e0000/evaluation_proposals.pt').stat().st_mtime_ns
    labels=json.loads((args.out/'e0000/training_targets.json').read_text())
    scores={int(k):v for k,v in labels['discovery_candidate_scores'].items()}
    assert labels['chosen_max_new_position']==max(scores,key=lambda p:(*scores[p],-p))
    assert labels['chosen_ig_position']==max(scores,key=lambda p:(scores[p][2],scores[p][0],-p))
    root=torch.load(args.out/'e0000/root_stats.pt',weights_only=True)
    logits=torch.load(args.out/'e0000/root_logits.pt',weights_only=True)
    x=torch.load(args.out/'e0000/root_input.pt',weights_only=True)
    masked=result['context']['masked']; ix=[masked.index(p) for p in result['selection']['common']]
    for phase in ['discovery','evaluation']:
        proposals=torch.load(args.out/f'e0000/{phase}_proposals.pt',weights_only=True)
        for seed,canvas in proposals.items():
            torch.manual_seed(seed); _,ids=sampler(logits,temperature=1.,top_p=1.,top_k=0)
            expected=x.clone(); expected[masked]=ids; assert torch.equal(canvas,expected)
    for row in result['rows']:
        child=torch.load(args.out/f'e0000/p{row["position"]}_v{row["sampled_id"]}.pt',weights_only=True)
        expect=x.clone(); expect[row['position']]=row['sampled_id']; assert torch.equal(expect,child['input_ids'])
        expected=int(((root['confidence'][ix]<.95)&(child['statistics']['confidence'][ix]>=.95)).sum())
        assert row['new_tokens']==expected
    example_plots(result,changes,args.out/'plots')
    archive_sha=pack_context(args.out/'e0000',args.out/'e0000.tar')
    data.atomic(args.out/'summary.json',dict(status='PASS_REAL_RECORD_CPU_CHECK',source_inventory=inventory,
                prior_500_identities_and_seeds_match=True,contexts=1,physical_batches=calls,
                sequence_forwards=seqs,all_170_candidates=True,fresh_rows=40,proposal_replay=True,
                independent_count_check=True,ig_anchor=labels['chosen_ig_position'],
                max_new_anchor=labels['chosen_max_new_position'],archive_sha256=archive_sha,
                new_corpus_positive_claim=False,gpu_run=False))
    print((args.out/'summary.json').read_text(),flush=True)


if __name__=='__main__': main()
