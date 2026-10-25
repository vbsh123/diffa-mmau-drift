import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import soundfile as sf
import torch

from drift_study.audio_windows import projected_tokens,wav_header,window_ranges
from drift_study.data import load_audio,prompt_token_count
from drift_study.protocol import Prepared


class LongAudioTests(unittest.TestCase):
    def test_window_coverage_and_token_growth(self):
        for seconds in (30,31,60,61,125,600):
            total=seconds*16000+7
            ranges=window_ranges(total)
            self.assertEqual(ranges[0][0],0)
            self.assertEqual(ranges[-1][1],total)
            self.assertEqual(sum(b-a for a,b in ranges),total)
            self.assertTrue(all(b-a<=480000 for a,b in ranges))
            self.assertTrue(all(a[1]==b[0] for a,b in zip(ranges,ranges[1:])))
        self.assertEqual(projected_tokens(60*16000,16000)['total'],878)
        self.assertEqual(projected_tokens(90*16000,16000)['total'],1317)
        self.assertEqual(projected_tokens(90*16000,16000)['acoustic'],192)

    def test_long_wave_header_and_loader_do_not_truncate(self):
        # Deliberately exceeds the old24MiB hidden payload cutoff as well as30s.
        samples=61*192000
        audio=np.zeros(samples,dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'long.wav'
            sf.write(path,audio,192000,subtype='FLOAT')
            self.assertGreater(path.stat().st_size,24*1024*1024)
            with path.open('rb') as stream:
                header=wav_header(stream)
                self.assertLess(stream.tell(),1024)
            self.assertEqual(header.frames,samples)
            self.assertEqual(header.samplerate,192000)
            result=load_audio(path)
            self.assertEqual(len(result),61*16000)

    def test_interleaved_acoustic_and_semantic_groups(self):
        prepared=Prepared(torch.ones(1,8),torch.ones(1,15,4),3,4,31,
            (dict(acoustic_tokens=2,semantic_tokens=5),dict(acoustic_tokens=2,semantic_tokens=6)))
        groups=prepared.groups()
        self.assertEqual(groups['acoustic'],[3,4,10,11])
        self.assertEqual(groups['semantic'],[5,6,7,8,9,12,13,14,15,16,17])
        self.assertEqual(sorted(p for g in groups.values() for p in g),list(range(prepared.prefix_length)))

    def test_estimated_tokens_match_real_feature_extractor_mask(self):
        from transformers import WhisperFeatureExtractor
        extractor=WhisperFeatureExtractor(feature_size=80)
        for seconds in (31.125,65.,119.999):
            samples=int(seconds*16000)
            measured=0
            for lo,hi in window_ranges(samples):
                inputs=extractor(np.zeros(hi-lo,dtype=np.float32),sampling_rate=16000,
                    return_attention_mask=True,return_tensors='pt',truncation=False)
                valid_frames=int(inputs.attention_mask.sum())
                hidden_frames=(valid_frames-1)//2+1
                measured+=hidden_frames//4+64
            expected=projected_tokens(samples,16000)
            self.assertEqual(measured,expected['total'])


@unittest.skipUnless(os.environ.get('DIFFA_ROOT'),'Set DIFFA_ROOT to pinned official checkout')
class NativeWindowEncodingTests(unittest.TestCase):
    def test_native_diffa_preserves_complete_long_audio_through_fpd(self):
        from drift_study.checkout import verify_checkout
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(verify_checkout(os.environ['DIFFA_ROOT'])))
        from src.configuration_DIFFA import DIFFAConfig
        from src.modeling_DIFFA import DIFFAModel
        from src.dataloader import build_chat_prompt
        from transformers import WhisperConfig,WhisperFeatureExtractor
        from drift_study.model import prepare
        class Tokenizer:
            eos_token='<eos>'
            def __call__(self,text,return_tensors=None):
                ids=[1]*len(text)
                if return_tensors:
                    return SimpleNamespace(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long))
                return {'input_ids':ids}
        tokenizer=Tokenizer()
        whisper=WhisperConfig(d_model=16,encoder_layers=3,decoder_layers=1,
            encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,
            decoder_ffn_dim=32,num_mel_bins=80,max_source_positions=1500)
        config=DIFFAConfig(whisper_config=whisper.to_dict(),llm_config={'d_model':16})
        config.whisper_model_id='openai/whisper-tiny'
        model=DIFFAModel(config,tokenizer=tokenizer,stage=2).eval()
        model.llm_model=SimpleNamespace(device=torch.device('cpu'),
            config=SimpleNamespace(max_sequence_length=4096,mask_token_id=31))
        extractor=WhisperFeatureExtractor(feature_size=80)
        torch.set_num_threads(2)
        audio=(load_audio(os.environ['REAL_LONG_AUDIO']) if os.environ.get('REAL_LONG_AUDIO')
               else np.sin(np.arange(65*16000,dtype=np.float32)*.01)*.1)
        prepared=prepare(model,tokenizer,extractor,audio,'Q?',128)
        estimate=projected_tokens(len(audio),16000)
        self.assertEqual(len(prepared.audio_chunks),estimate['windows'])
        self.assertEqual(prepared.audio_chunks[0]['start_sample'],0)
        self.assertEqual(prepared.audio_chunks[-1]['end_sample'],len(audio))
        self.assertEqual(sum(c['end_sample']-c['start_sample'] for c in prepared.audio_chunks),len(audio))
        self.assertEqual(prepared.speech.shape[1],estimate['total'])
        self.assertEqual(len(prepared.groups()['acoustic']),64*estimate['windows'])
        self.assertEqual(len(prepared.groups()['semantic']),estimate['semantic'])
        self.assertEqual(prepared.prompt.shape[1],prompt_token_count(tokenizer,'Q?'))
        # Shared formatting is byte-for-byte equal to the upstream chat builder.
        native=build_chat_prompt([{'role':'system','content':'You are a helpful voice assistant.'},
            {'role':'user','content':'This is the audio: <audio>. Q?'}],add_generation_prompt=True)
        self.assertEqual(prompt_token_count(tokenizer,'Q?'),len(native))
        # A short input still reproduces the original one-window features exactly.
        short_audio=audio[:20*16000]
        short=prepare(model,tokenizer,extractor,short_audio,'Q?',128)
        inputs=extractor(short_audio,sampling_rate=16000,return_tensors='pt',
                         return_attention_mask=True,truncation=False)
        with torch.no_grad():
            original,_=model.get_speech_features(inputs.input_features,inputs.attention_mask,stage=2)
        self.assertTrue(torch.equal(short.speech,original.bfloat16()))
        # Feed every projected long-audio token through the unchanged official FPD
        # using a tiny deterministic logit fixture, not full checkpoint QA weights.
        from drift_study.fpd import generate
        class LogitFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding=torch.nn.Embedding(32,16).bfloat16()
                self.seen_lengths=[]
            def get_input_embeddings(self):
                return self.embedding
            def forward(self,inputs_embeds):
                self.seen_lengths.append(inputs_embeds.shape[1])
                segment=inputs_embeds[:,prepared.insertion:prepared.insertion+prepared.speech.shape[1]]
                torch.testing.assert_close(segment,prepared.speech,atol=0,rtol=0)
                logits=torch.zeros(1,inputs_embeds.shape[1],32,dtype=torch.bfloat16)
                logits[:,:,5]=20.
                return SimpleNamespace(logits=logits)
        fixture=LogitFixture()
        output,nfe,_=generate(SimpleNamespace(llm_model=fixture,device=torch.device('cpu')),prepared)
        self.assertEqual(nfe,4)
        self.assertEqual(fixture.seen_lengths,[prepared.prefix_length+128]*4)
        self.assertEqual(output.shape[1],prepared.prompt.shape[1]+128)


if __name__=='__main__':
    unittest.main()
