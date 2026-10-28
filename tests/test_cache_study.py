from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from cache_study.engine import PrefixCache
from cache_study.protocol import (POLICIES,benchmark_source_digest,calibrate,parse_layers,start_from_mean,
                                  layer_schedule,fixed_start_settings,pyramid_specs)
from cache_study.run import measure,parser
from drift_study.protocol import prompt_for
from drift_study.storage import digest, source_digest, write_json
import test_study as reference_tests


class CacheProtocolTests(unittest.TestCase):
    def test_pyramid_depths_ratios_controls_and_synchronized_refreshes(self):
        specs=pyramid_specs()
        self.assertEqual(len(specs),6)
        for name,expected in [('pyramid_equal',[(64,64,64),(32,32,32),(16,16,16),(8,8,8)]),
                              ('pyramid_audio_slow',[(128,64,128),(64,32,64),(32,16,32),(16,8,16)]),
                              ('pyramid_audio_fast',[(32,64,32),(16,32,16),(8,16,8),(4,8,4)])]:
            schedule=specs[name]['layer_intervals']
            for layer in range(32):self.assertEqual(schedule[layer],expected[layer//8])
            # Every shallower refresh must also refresh the same modality deeper.
            for offset in range(1,257):
                for group in range(3):
                    for shallow,deep in [(0,8),(8,16),(16,24)]:
                        if offset%schedule[shallow][group]==0:
                            self.assertEqual(offset%schedule[deep][group],0)
        self.assertEqual(specs['888_deepsem2']['layer_intervals'][23],(8,8,8))
        self.assertEqual(specs['888_deepsem2']['layer_intervals'][24],(8,8,4))
        self.assertEqual(specs['16816']['layer_intervals'][31],(16,8,16))
    def test_requested_schedules_and_depth_selection(self):
        self.assertEqual(list(POLICIES.values()),[(16,16,16),(8,8,8),(16,8,16),(16,16,8),(32,32,16),(32,16,32)])
        self.assertEqual(parse_layers('16-31'),list(range(16,32)))
        self.assertEqual(parse_layers('16,20,24'),[16,20,24])
        self.assertEqual(parse_layers('all'),list(range(32)))
        for invalid in ('','32','24-16','-1','1,foo'):
            with self.assertRaises(ValueError):
                parse_layers(invalid)
        self.assertEqual(start_from_mean(25.5,'mean'),26)
        self.assertEqual(start_from_mean(25.5,'half-mean'),13)
        self.assertEqual(start_from_mean(25.5,'immediate'),0)

    def test_complete50_calibration_identity_and_partial_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            audio=root/'clip.wav'
            sf.write(audio,np.zeros(1600),16000)
            rows=[dict(id=str(i),audio='clip.wav',audio_sha256=digest(audio),audio_seconds=.1,
                domain=('speech','sound','music')[i%3],question='Q?',choices=['no','yes'],target='B',
                prompt=prompt_for('Q?',['no','yes'])) for i in range(50)]
            manifest=dict(dataset='test',revision='pinned',settings=dict(samples=50),rows=rows)
            path=root/'manifest.json'
            write_json(path,manifest)
            run=root/'run'
            write_json(run/'config.json',dict(manifest_sha256=digest(path),dataset='test',dataset_revision='pinned',
                sample_ids=[r['id'] for r in rows],source_sha256=source_digest(),seed=20261025,
                decoder=dict(max_answer_tokens=128,block_tokens=32,factor=1.,temperature=0.),models={}))
            for i,row in enumerate(rows):
                directory=run/'samples'/f'{i:05d}'
                directory.mkdir(parents=True)
                trace=directory/'trace.jsonl'
                trace.write_text('{}\n'*(i+1))
                write_json(directory/'result.json',dict(id=row['id'],audio_sha256=row['audio_sha256'],
                    forward_count=i+1,artifact_sha256={'trace.jsonl':digest(trace)}))
            result=calibrate(run,path,manifest)
            self.assertEqual(result['mean_forwards'],25.5)
            self.assertEqual(result['dense_warmup_forwards'],26)
            self.assertEqual(result['first_reuse_zero_based_forward'],26)
            missing=run/'samples'/'00049'/'result.json'
            missing.unlink()
            with self.assertRaisesRegex(ValueError,'all 50 completed'):
                calibrate(run,path,manifest)

    def test_forward10_and_exact_deep_semantic_profiles(self):
        args=parser().parse_args([])
        self.assertEqual((args.start,args.start_forward,args.deep_semantic_factor,args.deep_semantic_layers),
                         ('fixed',10,2,'24-31'))
        self.assertIsNone(fixed_start_settings(10)['mean_forwards'])
        self.assertEqual(fixed_start_settings(10)['first_reuse_zero_based_forward'],10)
        for base,expected in zip(POLICIES.values(),[(16,16,8),(8,8,4),(16,8,8),(16,16,4),(32,32,8),(32,16,16)]):
            schedule=layer_schedule(base,list(range(32)),list(range(24,32)),2)
            for layer in range(32):
                self.assertEqual(schedule[layer],expected if layer>=24 else base)


@unittest.skipUnless(os.environ.get('DIFFA_ROOT'),'Set DIFFA_ROOT to pinned official checkout')
class CacheEngineTests(unittest.TestCase):
    def setUp(self):
        reference_tests.OfficialFPDTests.setUp(self)

    def install_dense_reference(self,llm,intervals,layers,start,layer_intervals=None):
        banks={}
        clock=[0]
        groups=[self.prepared.groups()[g] for g in ('acoustic','text','semantic')]
        anchor=max(0,start-1)
        for layer,block in enumerate(llm.model.transformer.blocks):
            if layer not in layers:
                continue
            attention=block._scaled_dot_product_attention
            chosen=layer_intervals[layer] if layer_intervals is not None else intervals
            def inactive(chosen=chosen):
                return [p for group,interval in zip(groups,chosen)
                        if clock[0]>anchor and (clock[0]-anchor)%interval for p in group]
            def attend(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,layer=layer,attention=attention,inactive=inactive):
                skipped=inactive()
                if skipped:
                    k,v=k.clone(),v.clone()
                    k[:,:,skipped]=banks[layer][0][:,:,skipped]
                    v[:,:,skipped]=banks[layer][1][:,:,skipped]
                previous=banks[layer][2] if layer in banks else None
                banks[layer]=[k.clone(),v.clone(),previous]
                return attention(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal)
            block._scaled_dot_product_attention=attend
            def hidden(module,args,output,layer=layer,inactive=inactive):
                current=output[0].clone()
                skipped=inactive()
                if skipped:
                    current[:,skipped]=banks[layer][2][:,skipped]
                banks[layer][2]=current.clone()
                return current,output[1]
            block.register_forward_hook(hidden)
        return clock

    @torch.no_grad()
    def test_pyramid_shallow_deep_logits_against_independent_dense_stale_reference(self):
        for name in ('pyramid_equal','pyramid_audio_slow','pyramid_audio_fast'):
            spec=pyramid_specs()[name]
            # Exercise extrema of the full32-block schedule on the two-block model.
            schedule={0:spec['layer_intervals'][0],1:spec['layer_intervals'][31]}
            tested=deepcopy(self.llm).float();reference=deepcopy(self.llm).float()
            clock=self.install_dense_reference(reference,spec['base'],[0,1],10,schedule)
            with PrefixCache(tested,self.prepared,spec['base'],[0,1],10,schedule) as cache:
                for step in range(139):
                    clock[0]=step;torch.manual_seed(1200+step)
                    inputs=torch.randn(1,self.prepared.prefix_length+4,16)
                    torch.testing.assert_close(tested(inputs_embeds=inputs).logits,
                        reference(inputs_embeds=inputs).logits,atol=3e-6,rtol=3e-6)
                    for layer in (0,1):
                        self.assertTrue(all(p in cache.active_ids_by_layer[layer]
                            for p in range(self.prepared.prefix_length,inputs.shape[1])))
            for layer in (0,1):
                for i,group in enumerate(('acoustic','text','semantic')):
                    expected=sum(step<=9 or (step-9)%schedule[layer][i]==0 for step in range(139))
                    self.assertEqual(cache.accounting()['refresh_counts_by_layer'][str(layer)][group],expected)

    @torch.no_grad()
    def test_row_skipping_matches_independent_dense_stale_state_reference(self):
        # Every requested policy, immediate and delayed starts, all/deep-only layers.
        for intervals in POLICIES.values():
            for layers in ([0,1],[1]):
                for start in (0,3):
                    tested=deepcopy(self.llm).float()
                    reference=deepcopy(self.llm).float()
                    clock=self.install_dense_reference(reference,intervals,layers,start)
                    q_rows={0:[],1:[]}
                    handles=[]
                    for layer,block in enumerate(tested.model.transformer.blocks):
                        handles.append(block.q_proj.register_forward_pre_hook(
                            lambda module,args,layer=layer:q_rows[layer].append(args[0].shape[1])))
                    length=self.prepared.prefix_length+4
                    with PrefixCache(tested,self.prepared,intervals,layers,start) as cache:
                        for step in range(40):
                            clock[0]=step
                            torch.manual_seed(100+step)
                            inputs=torch.randn(1,length,16)
                            actual=tested(inputs_embeds=inputs).logits
                            expected=reference(inputs_embeds=inputs).logits
                            torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-6)
                            for layer in layers:
                                active=cache.active_ids_by_layer[layer]
                                self.assertTrue(all(i in active for i in range(self.prepared.prefix_length,length)))
                    for handle in handles:
                        handle.remove()
                    accounting=cache.accounting()
                    self.assertEqual(accounting['forwards'],40)
                    self.assertGreater(accounting['actual_reuse_forwards'],0)
                    self.assertGreater(accounting['skipped_token_update_fraction'],0)
                    if layers==[1]:
                        self.assertEqual(q_rows[0],[length]*40)
                        self.assertEqual(accounting['per_layer_computed_rows']['0'],length*40)
                    self.assertTrue(any(n<length for n in q_rows[1]))
                    self.assertEqual(q_rows[1][:max(1,start)],[length]*max(1,start))

    @torch.no_grad()
    def test_late_start_is_exact_dense_when_answer_finishes_first(self):
        reference=deepcopy(self.llm)
        with PrefixCache(self.llm,self.prepared,(16,8,16),[1],50) as cache:
            for step in range(5):
                inputs=torch.randn(1,self.prepared.prefix_length+4,16).bfloat16()
                self.assertTrue(torch.equal(self.llm(inputs_embeds=inputs).logits,
                                            reference(inputs_embeds=inputs).logits))
        self.assertEqual(cache.accounting()['actual_reuse_forwards'],0)
        self.assertEqual(cache.banks,{})

    @torch.no_grad()
    def test_native_fpd_benchmark_and_identity_preserve_decoding(self):
        tokenizer=SimpleNamespace(eos_token_id=126081,decode=lambda ids,skip_special_tokens:'Final answer: A')
        with patch('torch.cuda.synchronize'):
            baseline=measure(self.model,self.prepared,tokenizer,None,[1],0)
            identity=measure(self.model,self.prepared,tokenizer,(1,1,1),[1],0)
            cached=measure(self.model,self.prepared,tokenizer,(16,8,16),[1],3)
        self.assertFalse(baseline['failed'])
        self.assertEqual(baseline['output']['canvas_ids'],identity['output']['canvas_ids'])
        self.assertEqual(baseline['forward_count'],identity['forward_count'])
        self.assertGreater(cached['accounting']['actual_reuse_forwards'],0)
        self.assertEqual(cached['accounting']['cache_layers'],[1])

    @torch.no_grad()
    def test_bfloat16_deep_layer_cache_matches_dense_reference(self):
        from drift_study.protocol import Prepared
        self.prepared=Prepared(torch.tensor([[1,2,3,4]]),torch.randn(1,11,16).bfloat16(),2,4,31,
            (dict(acoustic_tokens=2,semantic_tokens=3),dict(acoustic_tokens=2,semantic_tokens=4)))
        tested=deepcopy(self.llm)
        reference=deepcopy(self.llm)
        intervals=(32,16,32)
        clock=self.install_dense_reference(reference,intervals,[1],3)
        with PrefixCache(tested,self.prepared,intervals,[1],3):
            for step in range(40):
                clock[0]=step
                torch.manual_seed(step+500)
                inputs=torch.randn(1,self.prepared.prefix_length+4,16).bfloat16()
                torch.testing.assert_close(tested(inputs_embeds=inputs).logits,
                                           reference(inputs_embeds=inputs).logits,atol=.02,rtol=.02)

    @torch.no_grad()
    def test_layer_specific_semantic_refresh_and_forward10_against_dense_reference(self):
        for factor in (1,2):
            for base in POLICIES.values():
                tested=deepcopy(self.llm).float()
                reference=deepcopy(self.llm).float()
                schedule=layer_schedule(base,[0,1],[1],factor)
                clock=self.install_dense_reference(reference,base,[0,1],10,schedule)
                with PrefixCache(tested,self.prepared,base,[0,1],10,schedule) as cache:
                    for step in range(45):
                        clock[0]=step
                        torch.manual_seed(900+step)
                        inputs=torch.randn(1,self.prepared.prefix_length+4,16)
                        torch.testing.assert_close(tested(inputs_embeds=inputs).logits,
                            reference(inputs_embeds=inputs).logits,atol=3e-6,rtol=3e-6)
                account=cache.accounting()
                self.assertEqual(account['actual_reuse_forwards'],35-35//max(base))
                for layer in (0,1):
                    for index,group in enumerate(('acoustic','text','semantic')):
                        interval=schedule[layer][index]
                        expected=sum(step<=9 or (step-9)%interval==0 for step in range(45))
                        self.assertEqual(account['refresh_counts_by_layer'][str(layer)][group],expected)
                comparison=self.assertGreater if factor==2 else self.assertEqual
                comparison(account['refresh_counts_by_layer']['1']['semantic'],
                           account['refresh_counts_by_layer']['0']['semantic'])


class CacheReportTests(unittest.TestCase):
    def test_partial_pairs_failures_and_actual_reuse_are_reported(self):
        from cache_study.report import analyze
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            write_json(root/'config.json',dict(policies={'888':[8,8,8]},cache_layers=[16,17],
                group_order=['acoustic','text','semantic'],calibration=dict(mean_forwards=25.5,
                    start_mode='mean',dense_warmup_forwards=26)))
            def row(identity,correct,seconds,failed=False,reuse=0):
                return dict(id=identity,domain='speech',failed=failed,seconds=seconds,forward_count=30,
                    score=dict(correct=correct,parse_failed=failed),
                    output=None if failed else dict(visible_tokens=20),
                    accounting=dict(actual_reuse_forwards=reuse,skipped_token_update_fraction=.2))
            write_json(root/'samples'/'00000'/'baseline.json',row('0',True,4.))
            write_json(root/'samples'/'00000'/'888.json',row('0',True,2.,reuse=4))
            write_json(root/'samples'/'00001'/'baseline.json',row('1',True,4.))
            write_json(root/'samples'/'00001'/'888.json',row('1',False,10.,failed=True))
            summary=analyze(root)
            self.assertEqual(summary['888']['n'],2)
            self.assertEqual(summary['888']['accuracy'],.5)
            self.assertEqual(summary['888']['failures'],1)
            self.assertEqual(summary['888']['paired_latency_speedup'],2.)
            self.assertEqual(summary['888']['paired_successful_n'],1)
            self.assertEqual(summary['888']['examples_with_actual_reuse'],1)
            self.assertEqual(summary['888']['losses'],1)


if __name__ == '__main__':
    unittest.main()
