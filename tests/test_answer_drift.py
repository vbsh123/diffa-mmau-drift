import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from drift_study.answer import AnswerDriftRecorder,TransitionObservers,token_cosine_distance
from drift_study.attention import AttentionRecorder
from drift_study.fpd import generate
import test_study as reference_tests


class AnswerDistanceTests(unittest.TestCase):
    def test_zero_safe_per_position_distance(self):
        a=torch.tensor([[0.,0.],[1.,0.],[1.,0.]])
        b=torch.tensor([[0.,0.],[0.,1.],[-1.,0.]])
        torch.testing.assert_close(token_cosine_distance(a,b),torch.tensor([0.,1.,2.]))


@unittest.skipUnless(os.environ.get('DIFFA_ROOT'),'Set DIFFA_ROOT to pinned checkout')
class NativeAnswerDriftTests(unittest.TestCase):
    def setUp(self):
        reference_tests.OfficialFPDTests.setUp(self)

    @torch.no_grad()
    def test_joint_observation_exact_decoding_and_identity_controlled_answer_populations(self):
        reference,nfe,_=generate(self.model,self.prepared,trace=False)
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'samples'/'00000'
            with AnswerDriftRecorder(self.llm,self.prepared,directory) as answer,AttentionRecorder(
                    self.llm,self.prepared,directory,every=4,query_chunk=32) as attention:
                output,observed_nfe,trace=generate(self.model,self.prepared,
                    observer=TransitionObservers(answer,attention))
            self.assertTrue(torch.equal(output,reference));self.assertEqual(nfe,observed_nfe)
            answer.finalize(40);attention.finalize()
            rows=[json.loads(line) for line in (directory/'answer_drift.jsonl').read_text().splitlines()]
            self.assertTrue(rows);self.assertTrue(all(r['step']>0 and r['count']<=40 for r in rows))
            ids=torch.full((128,),self.prepared.mask_id);history=[ids.clone()]
            for item in trace:
                ids=ids.clone();ids[item['answer_positions']]=torch.tensor(item['token_ids']);history.append(ids)
            for row in rows:
                current=history[row['step']][:40];old=history[row['step']-1 if row['reference']=='previous' else 0][:40]
                same=current==old
                expected={'answer_content_all':40,'answer_content_masked_stable':int((same&(current==self.prepared.mask_id)).sum()),
                    'answer_content_decoded_stable':int((same&(current!=self.prepared.mask_id)).sum()),'answer_content_changed':int((~same).sum())}
                self.assertEqual(row['count'],expected[row['group']])
            partition={}
            for row in rows:
                partition.setdefault((row['step'],row['layer'],row['reference']),{})[row['group']]=row
            for group in partition.values():
                all_row=group['answer_content_all']
                controls=[r for name,r in group.items() if name!='answer_content_all']
                self.assertEqual(sum(r['count'] for r in controls),all_row['count'])
                reconstructed=sum(r['count']*r['metrics']['mean'] for r in controls)/all_row['count']
                self.assertAlmostEqual(reconstructed,all_row['metrics']['mean'],places=6)
            self.assertTrue(any(r['group']=='answer_content_decoded_stable' for r in rows))
            self.assertFalse(any(r['reference']=='initial' and r['group']=='answer_content_decoded_stable' for r in rows))
            self.assertTrue(any(r['group']=='answer_content_changed' for r in rows))
            # Observer hooks are removed and do not affect the next clean decode.
            again,again_nfe,_=generate(self.model,self.prepared,trace=False)
            self.assertTrue(torch.equal(again,reference));self.assertEqual(again_nfe,nfe)
            # A complete sample exports answer drift in HTML without a model load.
            (directory/'result.json').write_text(json.dumps(dict(id='fixture',domain='speech',forward_count=nfe)))
            (directory/'drift.jsonl').write_text('')
            from drift_study.answer_report import collect,build
            data=collect(Path(temporary));self.assertTrue(data['examples'][0]['answer_available'])
            path=build(temporary);self.assertTrue(path.exists());self.assertIn('Stable decoded answer',path.read_text())


if __name__=='__main__':unittest.main()
