import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
import native_protocol as protocol
from run import require_a100,pack_context
from report import ARMS,summarize


class ProtocolTest(unittest.TestCase):
    def test_exactly_one_a100(self):
        def mock(n,name):
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:n>0,device_count=lambda:n,
                                                       get_device_name=lambda index:name))
        self.assertEqual(require_a100(mock(1,'NVIDIA A100-SXM4-80GB')),'NVIDIA A100-SXM4-80GB')
        for n,name in [(0,'A100'),(2,'A100'),(1,'NVIDIA H100'),(1,'NVIDIA L40S')]:
            with self.assertRaises(RuntimeError): require_a100(mock(n,name))

    def test_new_not_total_and_revealed_excluded(self):
        root=dict(confidence=torch.tensor([.2,.96,.3,.2]),entropy=torch.tensor([2.,.2,2.,2.]),
                  reference_correct=torch.tensor([True,True,True,True]),reference_probability=torch.tensor([.2,.96,.3,.2]))
        child=dict(confidence=torch.tensor([1.,.97,.96,.1]),entropy=torch.tensor([0.,.1,.2,2.1]),
                   reference_correct=torch.tensor([True,True,True,False]),reference_probability=torch.tensor([1.,.97,.96,.05]))
        m=protocol.metrics(root,child,[1,2,3])
        self.assertEqual(m['new_tokens'],1)
        self.assertEqual(m['new_indices'],[2])
        self.assertEqual(m['new_correct'],1)
        self.assertEqual(m['correct_after'],2)
        self.assertEqual(m['new_confident_correct'],1)
        self.assertEqual(m['new_confident_wrong'],0)

    def test_lost_and_negative_ig_retained(self):
        root=dict(confidence=torch.tensor([.99]),entropy=torch.tensor([.1]),reference_correct=torch.tensor([True]),reference_probability=torch.tensor([.99]))
        child=dict(confidence=torch.tensor([.3]),entropy=torch.tensor([1.9]),reference_correct=torch.tensor([True]),reference_probability=torch.tensor([.3]))
        m=protocol.metrics(root,child,[0])
        self.assertEqual(m['new_tokens'],0); self.assertEqual(m['net_tokens'],-1)
        self.assertLess(m['entropy_drop'],0)

    def test_independent_teacher_rankings_and_ties(self):
        scores={1:(2.,2.,.1),2:(1.,1.,8.),3:(2.,2.,.1)}
        count=max(scores,key=lambda p:(*scores[p],-p))
        ig=max(scores,key=lambda p:(scores[p][2],scores[p][0],-p))
        self.assertEqual(count,1); self.assertEqual(ig,2)

    def test_archive_retains_training_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='dna-cache-test-') as tmp:
            root=Path(tmp); context=root/'context'; context.mkdir()
            (context/'training_targets.json').write_text(json.dumps({'chosen_ig_position':3,'chosen_max_new_position':8}))
            torch.save(torch.tensor([1,2,3]),context/'root_input.pt')
            digest=pack_context(context,root/'cache.tar')
            self.assertEqual(len(digest),64); self.assertTrue((context/'root_input.pt').exists())

    def test_all_outcome_report_and_compact_replay(self):
        with tempfile.TemporaryDirectory(prefix='dna-report-test-') as tmp:
            state=Path(tmp); part=state/'results'/'part0000'; part.mkdir(parents=True)
            (state/'contract.json').write_text(json.dumps({'expected_contexts':1}))
            rows=[]
            for arm in ARMS:
                for draw in range(8):
                    new=int(arm=='max_new' and draw==0)
                    rows.append(dict(arm=arm,new_tokens=new,new_tokens_090=new,new_correct=new,
                                     new_confident_correct=new,new_confident_wrong=0,
                                     new_wrong=0,entropy_drop=-1.,max_probability_gain=.2,
                                     reference_probability_gain=.1,unlocked=[{'token':'ACGTAC'}]*new))
            (part/'e000000.json').write_text(json.dumps(dict(rows=rows,context={'index':0},
                dataset={'in_original_four_species':True,'development':True})))
            first=summarize(state); second=summarize(state)
            self.assertEqual(first,second)
            self.assertEqual(first['mean_new_tokens']['max_new'],1/8)
            self.assertEqual(first['arms']['max_new']['ig_sum'],-8.)
            self.assertTrue((state/'plots'/'unlocked_vs_probability_shootup.png').exists())
            self.assertTrue((state/'plot_statistics'/'part0000'/'e000000.json').exists())


if __name__=='__main__': unittest.main()
