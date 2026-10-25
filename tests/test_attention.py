import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from drift_study.attention import AttentionRecorder,group_attention_mass
from drift_study.fpd import generate
import test_study as reference_tests


class AttentionMathTests(unittest.TestCase):
    def test_chunked_modality_mass_matches_independent_dense_softmax(self):
        torch.manual_seed(17)
        q=torch.randn(1,3,11,4)
        k=torch.randn_like(q)
        groups=[[0,4],[1,2,5,8],[3],[6,7],[9,10]]
        masks=torch.zeros(5,11,dtype=torch.bool)
        for i,indices in enumerate(groups):
            masks[i,indices]=True
        mass,relative,counts=group_attention_mass(q,k,masks,query_chunk=3)
        dense=(q@k.transpose(-1,-2)/2).softmax(-1)[0]
        expected=torch.stack([torch.stack([dense[:,qs][:,:,ks].sum(-1).mean()
            for ks in groups]) for qs in groups])
        torch.testing.assert_close(mass,expected,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(mass.sum(-1),torch.ones(5),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(relative,expected/(counts[None,:]/11),atol=1e-6,rtol=1e-6)


@unittest.skipUnless(os.environ.get('DIFFA_ROOT'),'Set DIFFA_ROOT to pinned official checkout')
class NativeAttentionTests(unittest.TestCase):
    def setUp(self):
        reference_tests.OfficialFPDTests.setUp(self)

    @torch.no_grad()
    def test_attention_observation_preserves_fpd_and_tracks_answer_groups(self):
        reference,nfe,_=generate(self.model,self.prepared,trace=False)
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'samples'/'00000'
            with AttentionRecorder(self.llm,self.prepared,directory,every=4,query_chunk=32) as recorder:
                output,observed_nfe,_=generate(self.model,self.prepared,observer=recorder)
            recorder.finalize()
            self.assertTrue(torch.equal(reference,output))
            self.assertEqual(nfe,observed_nfe)
            rows=[json.loads(line) for line in (directory/'attention.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),len(range(0,nfe,4))*2)
            first=rows[0]
            self.assertEqual(first['counts']['answer_masked'],128)
            self.assertEqual(first['counts']['answer_decoded'],0)
            self.assertNotIn('answer_decoded',first['mass'])
            self.assertGreater(rows[-1]['counts']['answer_decoded'],0)
            for row in rows:
                self.assertEqual(sum(row['counts'].values()),self.prepared.prefix_length+128)
                for keys in row['mass'].values():
                    self.assertAlmostEqual(sum(keys.values()),1.,places=5)
            (directory/'result.json').write_text('{}')
            from drift_study.attention_report import analyze
            summary=analyze(temporary,plots=True)
            self.assertEqual(summary['examples'],1)
            self.assertEqual(summary['observed_layer_forward_records'],len(rows))
            self.assertTrue((Path(temporary)/'analysis'/'attention_answer_decoded.png').exists())


if __name__=='__main__':
    unittest.main()
