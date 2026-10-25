import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch

from drift_study.data import metadata_eligible, prepare, quotas, read_manifest
from drift_study.fpd import generate, output_record
from drift_study.metrics import distances, HiddenRecorder, pairwise_matrices
from drift_study.protocol import Prepared, grade, prompt_for
from drift_study.remote_zip import HTTPRangeFile
from drift_study.storage import write_json


class ProtocolTests(unittest.TestCase):
    def test_noncontiguous_text_and_audio_membership(self):
        prepared = Prepared(torch.ones(1, 8), torch.ones(1, 7, 16), 3, 2, 31)
        self.assertEqual(prepared.groups(), dict(acoustic=[3,4], semantic=[5,6,7,8,9], text=[0,1,2,10,11,12,13,14]))
        all_positions = [p for values in prepared.groups().values() for p in values]
        self.assertEqual(sorted(all_positions), list(range(15)))

    def test_reasoning_prompt_and_conservative_final_grading(self):
        text = prompt_for('What happened?', ['One', 'Two'])
        self.assertIn('Think step by step', text)
        self.assertIn('A. One\nB. Two', text)
        self.assertTrue(grade('First A seems possible.\nFinal answer: B', 'B')['correct'])
        self.assertTrue(grade('Final answer: A\nFinal answer: B.', 'B')['correct'])
        for text in ('A sound is heard.', 'Final answer: B because ...', 'Final answer: unknown'):
            self.assertTrue(grade(text, 'B')['parse_failed'])

    def test_quotas_and_metadata_eligibility(self):
        self.assertEqual(quotas(50), dict(speech=17, sound=17, music=16))
        row = dict(id='a', category='speech', length_type='short', audio_path=['data/a.wav'], question='Q', choices=['A','B'], answer='B')
        self.assertTrue(metadata_eligible(row))
        self.assertTrue(metadata_eligible(dict(row,length_type='long')))
        self.assertTrue(metadata_eligible(dict(row,length_type='ultra-long')))
        for change in (dict(category='voice_chat'), dict(audio_path=['a','b']),
                       dict(choices=['A','A']), dict(answer='C'), dict(id='../unsafe')):
            self.assertFalse(metadata_eligible(dict(row, **change)))


class RangeTests(unittest.TestCase):
    def test_seek_and_ranges_and_full_download_refusal(self):
        payload = b'0123456789'
        requests = []
        class Response(io.BytesIO):
            status = 206
        def open_range(request, timeout):
            header = request.get_header('Range')
            start, end = map(int, header.removeprefix('bytes=').split('-'))
            requests.append((start,end))
            response = Response(payload[start:end+1])
            response.headers = {'Content-Range':f'bytes {start}-{end}/{len(payload)}'}
            return response
        with patch('drift_study.remote_zip.urlopen', side_effect=open_range):
            with HTTPRangeFile('https://example.invalid/data.zip') as remote:
                remote.seek(-3, 2)
                self.assertEqual(remote.read(3), b'789')
                self.assertEqual(remote.read(), b'')
                remote.seek(1)
                self.assertEqual(remote.read(2), b'12')
                self.assertEqual(remote.bytes_received, 6)
        self.assertEqual(requests, [(0,0),(7,9),(1,2)])
        response = Response(b'')
        response.status, response.headers = 200, {}
        with patch('drift_study.remote_zip.urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'refusing full archive'):
                HTTPRangeFile('https://example.invalid/archive')


class DataTests(unittest.TestCase):
    def test_seeded_selection_real_wav_headers_and_integrity(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as target:
                for domain in ('speech','sound','music'):
                    for i in range(3):
                        identity = f'{domain}_{i}'
                        audio = io.BytesIO()
                        wave = np.linspace(-.2,.2,1600,dtype=np.float32)
                        sf.write(audio, wave, 16000, format='WAV')
                        target.writestr('data/'+identity+'.wav', audio.getvalue())
                        rows.append(dict(id=identity, category=domain, length_type='short',
                            audio_path=['data/'+identity+'.wav'], question='Question?',
                            choices=['no','yes'], answer='yes'))
            metadata = root/'metadata.parquet'
            pq.write_table(pa.Table.from_pylist(rows), metadata)
            class FakeRemote(io.BytesIO):
                def __init__(self, url):
                    super().__init__(archive.getvalue())
                    self.bytes_received = len(archive.getvalue())
            with patch('huggingface_hub.HfApi') as api, \
                 patch('huggingface_hub.hf_hub_download', return_value=str(metadata)), \
                 patch('huggingface_hub.hf_hub_url', return_value='unused'), \
                 patch('drift_study.data.context_spec',return_value=(lambda text:{'input_ids':[1]*20},4096)), \
                 patch('drift_study.data.HTTPRangeFile', FakeRemote):
                api.return_value.dataset_info.return_value.sha = 'pinned'
                one = prepare(root/'one'/'manifest.json',6,revision='pinned',min_seconds=0)
                two = prepare(root/'two'/'manifest.json',6,revision='pinned',min_seconds=0)
            self.assertEqual([r['id'] for r in one['rows']], [r['id'] for r in two['rows']])
            self.assertEqual([r['domain'] for r in one['rows'][:3]], ['speech','sound','music'])
            self.assertEqual(one['quotas'], dict(speech=2,sound=2,music=2))
            self.assertEqual(read_manifest(root/'one'/'manifest.json')['rows'], one['rows'])
            for row in one['rows']:
                self.assertAlmostEqual(row['audio_seconds'], .1)
                self.assertEqual(row['target'], 'B')
            (root/'one'/one['rows'][0]['audio']).write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                read_manifest(root/'one'/'manifest.json')


class MetricTests(unittest.TestCase):
    def test_token_mean_and_flattened_are_distinct_and_zero_safe(self):
        a = torch.tensor([[100.,0.],[0.,1.]])
        b = torch.tensor([[100.,0.],[1.,0.]])
        result = distances(a,b)
        self.assertAlmostEqual(result['mean'], .5)
        self.assertLess(result['flattened'], .001)
        self.assertEqual(distances(torch.zeros(2,3),torch.zeros(2,3))['mean'], 0.)
        self.assertEqual(distances(torch.zeros(2,3),torch.ones(2,3))['mean'], 1.)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            distances(torch.full((2,3), float('nan')), torch.ones(2,3))

    def test_disk_pairwise_matches_independent_per_token_reference(self):
        torch.manual_seed(7)
        states = [torch.randn(7,5) for _ in range(4)]
        states[0][2] = 0
        states[1][2] = 0
        positions = [0,2,4,6]
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for i,state in enumerate(states):
                path=Path(temporary)/f'{i}.npy'
                np.save(path, state.numpy())
                paths.append(path)
            token, flat = pairwise_matrices(paths, positions, chunk_tokens=2)
        for i in range(4):
            for j in range(4):
                expected = distances(states[i][positions],states[j][positions])
                self.assertAlmostEqual(float(token[i,j]),1-expected['mean'],places=6)
                self.assertAlmostEqual(float(flat[i,j]),1-expected['flattened'],places=6)


@unittest.skipUnless(os.environ.get('DIFFA_ROOT'), 'Set DIFFA_ROOT to pinned official checkout')
class OfficialFPDTests(unittest.TestCase):
    def setUp(self):
        from drift_study.checkout import verify_checkout
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(verify_checkout(os.environ['DIFFA_ROOT'])))
        from src.fast_dllm.model.configuration_llada import LLaDAConfig
        from src.fast_dllm.model.modeling_llada import LLaDAModelLM
        torch.set_num_threads(2)
        torch.manual_seed(23)
        config = LLaDAConfig(d_model=16,n_heads=2,n_kv_heads=2,n_layers=2,mlp_hidden_size=32,
            vocab_size=32,embedding_size=32,block_type='llama',block_group_size=1,rope=True,
            activation_type='silu',max_sequence_length=256,flash_attention=False,weight_tying=False,
            mask_token_id=31,eos_token_id=30,attention_dropout=0.,residual_dropout=0.,embedding_dropout=0.)
        self.llm = LLaDAModelLM(config,init_params=True).eval().to(torch.bfloat16)
        # Test fixture only: exclude MASK predictions so the random model terminates.
        def avoid_mask(module, args, output):
            output.logits[...,31] = -100
        self.llm.register_forward_hook(avoid_mask)
        self.model = SimpleNamespace(llm_model=self.llm,device=self.llm.device)
        self.prepared = Prepared(torch.tensor([[1,2,3,4]]),torch.randn(1,5,16).bfloat16(),2,2,31)

    @torch.no_grad()
    def test_observed_official_decoder_exact_ids_nfe_and_full_coverage(self):
        from drift_study.report import analyze
        from src.fast_dllm import diffa_generate as upstream
        original = upstream.get_transfer_index_dynamic
        reference, reference_nfe, _ = generate(self.model,self.prepared,trace=False)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'samples'/'00000'
            with HiddenRecorder(self.llm,self.prepared,directory,pairwise=True) as recorder:
                output,nfe,trace = generate(self.model,self.prepared)
            self.assertTrue(torch.equal(output,reference))
            self.assertEqual(nfe,reference_nfe)
            self.assertEqual(len(trace),nfe)
            self.assertIs(upstream.get_transfer_index_dynamic, original)
            self.assertEqual(sum(t['commits'] for t in trace),128)
            self.assertEqual({t['block'] for t in trace},{0,1,2,3})
            recorder.finalize()
            self.assertFalse((directory/'matrix_scratch').exists())
            measurement=json.loads((directory/'measurement.json').read_text())
            self.assertEqual(measurement['layers'],[0,1])
            self.assertEqual(measurement['groups'],dict(semantic=3,acoustic=2,text=4))
            records=[json.loads(s) for s in (directory/'drift.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),(nfe*2-1)*2*3)
            # Prefix states move even though prefix input token identities are fixed.
            self.assertTrue(any(r['metrics']['mean']>0 for r in records if r['reference']=='previous'))
            write_json(directory/'result.json',dict(id='toy',domain='speech',score=grade('Final answer: A','A'),
                output=dict(capacity_exhausted=True),forward_count=nfe))
            summary=analyze(Path(temporary),plots=True)
            self.assertEqual(summary['all']['n'],1)
            self.assertEqual(summary['all']['n_with_drift'],1)
            self.assertTrue((Path(temporary)/'analysis'/'previous_raw_drift.png').exists())

    def test_observer_restores_hooks_on_exception(self):
        initial = [len(b._forward_hooks) for b in self.llm.model.transformer.blocks]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with HiddenRecorder(self.llm,self.prepared,temporary):
                    raise RuntimeError('intentional')
        self.assertEqual([len(b._forward_hooks) for b in self.llm.model.transformer.blocks],initial)

    def test_official_factor_selects_variable_commit_counts(self):
        from src.fast_dllm.diffa_generate import get_transfer_index_dynamic
        x = torch.full((1,8),31)
        eligible = torch.ones_like(x,dtype=torch.bool)
        uncertain = torch.zeros(1,8,32)
        certain = uncertain.clone()
        certain[:,:,5] = 20.
        _, low = get_transfer_index_dynamic(uncertain,0.,'low_confidence',eligible,x,None,1.)
        _, high = get_transfer_index_dynamic(certain,0.,'low_confidence',eligible,x,None,1.)
        self.assertEqual(int(low.sum()),1)
        self.assertEqual(int(high.sum()),8)

    @torch.no_grad()
    def test_official_eos_waits_for_preceding_masks_and_stops_without_filling_canvas(self):
        prepared = Prepared(torch.tensor([[1,2]]),torch.zeros(1,2,2).bfloat16(),1,1,126336)
        class ScriptedLLM:
            def __init__(self):
                self.embedding = torch.nn.Embedding(126337,2).bfloat16()
                self.step = 0
            def get_input_embeddings(self):
                return self.embedding
            def __call__(self, inputs_embeds):
                logits = torch.zeros(1,inputs_embeds.shape[1],126337,dtype=torch.bfloat16)
                # Reveal EOS first, then resolve the token preceding it.
                position = prepared.prefix_length+(1 if self.step==0 else 0)
                token = 126081 if self.step==0 else 17
                logits[0,position,token] = 20.
                self.step += 1
                return SimpleNamespace(logits=logits)
        model = SimpleNamespace(llm_model=ScriptedLLM(),device=torch.device('cpu'))
        output,nfe,trace=generate(model,prepared)
        self.assertEqual(nfe,2)
        self.assertEqual(trace[0]['answer_positions'],[1])
        self.assertEqual(trace[1]['answer_positions'],[0])
        tokenizer=SimpleNamespace(eos_token_id=126081,decode=lambda ids,skip_special_tokens:str(ids))
        record=output_record(output,prepared,tokenizer)
        self.assertTrue(record['eos_found'])
        self.assertFalse(record['capacity_exhausted'])
        self.assertEqual(record['visible_ids'],[17])
        self.assertEqual(record['unresolved_visible_masks'],0)
        self.assertIn(prepared.mask_id,record['canvas_ids'][32:])


if __name__ == '__main__':
    unittest.main()
