"""GSM8K-style before/after position plots and both anchor teacher readouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import atomic

ARMS=['max_new','ig','confidence','random','earliest']


def compact_statistics(result):
    """Persist small sufficient statistics; never reread discovery tensors to plot."""
    assert len(result['rows'])==40
    stats={}
    for arm in ARMS:
        rows=[row for row in result['rows'] if row['arm']==arm]
        assert len(rows)==8
        stats[arm]=dict(samples=8,
            new=sum(r['new_tokens'] for r in rows),
            new90=sum(r['new_tokens_090'] for r in rows),
            any=sum(r['new_tokens']>0 for r in rows),
            two=sum(r['new_tokens']>=2 for r in rows),
            new_correct=sum(r['new_confident_correct'] for r in rows),
            new_wrong=sum(r['new_confident_wrong'] for r in rows),
            ig_sum=sum(r['entropy_drop'] for r in rows),
            max_probability_gain=sum(r['max_probability_gain'] for r in rows),
            reference_probability_gain=sum(r['reference_probability_gain'] for r in rows),
            non_sixmer_targets=sum(not(len(u['token'])==6 and set(u['token'])<=set('ACGT'))
                                  for r in rows for u in r['unlocked']))
    return dict(index=result['context']['index'],
                flags=[result['dataset']['in_original_four_species'],result['dataset']['development']],
                stats=stats)


def example_plots(result,changes,out):
    out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    pairs=[('before_max_probability','after_max_probability','Max token probability'),
           ('before_reference_probability','after_reference_probability','Reference token probability'),
           ('before_entropy_bits','after_entropy_bits','Entropy (bits)')]
    positions=changes['common_positions']
    for i,arm in enumerate(['max_new','ig']):
        ix=next(j for j,row in enumerate(result['rows']) if row['arm']==arm)
        row=result['rows'][ix]
        for j,(before,after,label) in enumerate(pairs):
            ax=axes[i,j]
            ax.plot(positions,changes[before][ix],label='Before',lw=.8)
            ax.plot(positions,changes[after][ix],label='After sampled anchor',lw=.8)
            ax.axvline(row['position'],color='black',ls='--',lw=.8,label='Anchor position')
            if j<2: ax.axhline(.95,color='gray',ls=':',lw=.8); ax.set_ylim(0,1)
            ax.set(xlabel='Token position (6 DNA bases per interior token)',ylabel=label,
                   title=f'{arm}: {row["token"]} at {row["position"]}; new={row["new_tokens"]}')
        axes[i,0].legend(fontsize=7)
    fig.suptitle(f'CSV row {result["context"]["row_index"]}: first fresh sample, not best sample')
    fig.tight_layout(); fig.savefig(out/'before_after_positions.png',dpi=150); plt.close(fig)
    positions=sorted(map(int,result['selection']['candidate_scores']))
    scores=[result['selection']['candidate_scores'][str(pos)] for pos in positions]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,idx,label in [(axes[0],2,'Mean IG (entropy reduction, bits)'),(axes[1],0,'Mean newly ≥95%-confident tokens')]:
        ax.plot(positions,[v[idx] for v in scores],'.-',lw=.5,ms=2)
        for arm,color in [('ig','tab:orange'),('max_new','tab:green')]:
            ax.axvline(result['selection']['actions'][arm],label=arm,color=color,ls='--')
        ax.set(xlabel='Candidate anchor position',ylabel=label); ax.legend()
    fig.suptitle('All 170 tested positions; two discovery samples each')
    fig.tight_layout(); fig.savefig(out/'ig_and_max_new_by_anchor.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,arm in zip(axes,['max_new','ig']):
        indices=[i for i,r in enumerate(result['rows']) if r['arm']==arm]
        before=changes['before_same_token_probability'][indices].ravel()
        after=changes['after_same_token_probability'][indices].ravel()
        ax.scatter(before,after,s=3,alpha=.15)
        ax.plot([0,1],[0,1],color='black',lw=.7); ax.axhline(.95,color='gray',ls=':')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Before: probability of child-predicted token',
               ylabel='After: probability of same token',title=f'{arm}: all 8 draws, all common targets')
    fig.tight_layout(); fig.savefig(out/'probability_shootup.png',dpi=150); plt.close(fig)


def summarize(state):
    state=Path(state); contract=json.loads((state/'contract.json').read_text())
    values=[]; gains=[]; strata=[]; contexts=[]
    stats={arm:dict(samples=0,new=0,new90=0,any=0,two=0,new_correct=0,new_wrong=0,
                   ig_sum=0.,max_probability_gain=0.,reference_probability_gain=0.,non_sixmer_targets=0)
           for arm in ARMS}
    for path in sorted((state/'results').glob('part*/e*.json')):
        compact=state/'plot_statistics'/path.parent.name/path.name
        if compact.exists():
            r=json.loads(compact.read_text())
        else:
            r=compact_statistics(json.loads(path.read_text()))
            compact.parent.mkdir(parents=True,exist_ok=True)
            atomic(compact,r)
        rowmeans=[]; rowgains=[]
        for arm in ARMS:
            s=r['stats'][arm]
            for key in stats[arm]: stats[arm][key]+=s[key]
            rowmeans.append(s['new']/s['samples'])
            rowgains.append(s['max_probability_gain']/s['samples'])
        values.append(rowmeans); gains.append(rowgains); strata.append(r['flags'])
        contexts.append(r['index'])
    if not values: return
    values=np.asarray(values); flags=np.asarray(strata,dtype=bool)
    report=dict(status='COMPLETE_MEASUREMENT' if len(values)==contract['expected_contexts'] else 'PARTIAL',
                contexts=len(values),expected=contract['expected_contexts'],arms=stats,
                mean_new_tokens={arm:float(values[:,i].mean()) for i,arm in enumerate(ARMS)},
                paired_against_confidence={arm:float((values[:,i]-values[:,2]).mean()) for i,arm in enumerate(ARMS[:2])},
                strata={},biological_validity_established=False,generation_speedup_measured=False,
                cache_is_singleton_offline_teacher_not_trained_policy=True)
    for name,col,flag in [('original_four_species',0,True),('other_species',0,False),
                          ('development',1,True),('newly_screened',1,False)]:
        sub=values[flags[:,col]==flag]
        report['strata'][name]=dict(contexts=len(sub),mean_new_tokens={arm:float(sub[:,i].mean()) if len(sub) else None
                                                                      for i,arm in enumerate(ARMS)})
    plots=state/'plots'; plots.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    metrics=[([report['mean_new_tokens'][a] for a in ARMS],'New ≥95%-confident other tokens / sampled anchor'),
             ([stats[a]['ig_sum']/stats[a]['samples'] for a in ARMS],'IG: entropy reduction / sampled anchor (bits)'),
             ([stats[a]['max_probability_gain']/stats[a]['samples'] for a in ARMS],'Summed max-probability increase / sampled anchor')]
    for ax,(numbers,label) in zip(axes,metrics):
        ax.bar(ARMS,numbers); ax.set_ylabel(label); ax.tick_params(axis='x',rotation=30)
    fig.suptitle(f'{len(values):,} sequences; all fresh outcomes, including zero and negative effects')
    fig.tight_layout(); fig.savefig(plots/'both_teachers_vs_controls.png',dpi=150); plt.close(fig)
    gains=np.asarray(gains)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,ax in enumerate(axes):
        h=ax.hexbin(gains[:,i],values[:,i],gridsize=40,mincnt=1,bins='log')
        ax.set(xlabel='Summed max-probability increase / sampled anchor',
               ylabel='New ≥95%-confident other tokens / sampled anchor',title=ARMS[i])
        fig.colorbar(h,ax=ax,label='Sequences (log color scale)')
    fig.suptitle('All sequences; means over 8 fresh draws per selected anchor')
    fig.tight_layout(); fig.savefig(plots/'unlocked_vs_probability_shootup.png',dpi=150); plt.close(fig)
    atomic(state/'summary.json',report)
    np.savez_compressed(state/'sequence_level_comparisons.npz',indices=np.asarray(contexts),means=values,
                        max_probability_gains=gains,arms=np.asarray(ARMS),flags=flags)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--state',type=Path,required=True)
    summarize(ap.parse_args().state)
