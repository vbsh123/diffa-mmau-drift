from contextlib import redirect_stdout
from copy import deepcopy
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
from scipy.stats import binomtest

from cache_study.paired import comparison,exact_mcnemar,clip_signflip,analyze
from cache_study.protocol import POLICIES,layer_schedule
from drift_study.audio_windows import INPUT_PROTOCOL
from drift_study.data import prepare,read_manifest,DATASET
from drift_study.protocol import prompt_for
from drift_study.storage import digest,write_json


class FreshCohortTests(unittest.TestCase):
    def test_fresh_unique_sampling_excludes_old_ids_members_and_duplicate_wavs(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);archive=io.BytesIO();metadata=[];old=[]
            with zipfile.ZipFile(archive,'w') as z:
                for di,domain in enumerate(('speech','sound','music')):
                    waves={}
                    for k in range(6):
                        # k0/k1 are old audio; k2/k3 are another duplicated WAV.
                        family=0 if k<2 else 2 if k<4 else k
                        value=(di*10+family+1)/100
                        shape=1600 if k<2 else (1600,2)
                        wave=io.BytesIO();sf.write(wave,np.full(shape,value,dtype=np.float32),16000,format='WAV')
                        payload=wave.getvalue();member=f'data/{domain}{k}.wav';z.writestr(member,payload)
                        metadata.append(dict(id=f'{domain}{k}',category=domain,audio_path=[member],
                            question='Q?',choices=['yes','no'],answer='yes'))
                        if k==0:
                            path=root/f'old_{domain}.wav';path.write_bytes(payload)
                            old.append(dict(id=f'{domain}{k}',audio=path.name,audio_sha256=digest(path),
                                source_audio=member,question='Q?',choices=['yes','no'],prompt=prompt_for('Q?',['yes','no'])))
            old_manifest=root/'old.json';write_json(old_manifest,dict(dataset=DATASET,revision='pinned',settings=dict(samples=3),rows=old))
            meta=root/'metadata.parquet';pq.write_table(pa.Table.from_pylist(metadata),meta)
            class Remote(io.BytesIO):
                def __init__(self,url):super().__init__(archive.getvalue());self.bytes_received=len(archive.getvalue())
            with patch('huggingface_hub.HfApi') as api,patch('huggingface_hub.hf_hub_download',return_value=str(meta)),\
                 patch('huggingface_hub.hf_hub_url',return_value='unused'),patch('drift_study.data.HTTPRangeFile',Remote),\
                 patch('drift_study.data.context_spec',return_value=(lambda text:{'input_ids':[1]*20},4096)):
                api.return_value.dataset_info.return_value.sha='pinned'
                result=prepare(root/'fresh.json',samples=6,revision='pinned',min_seconds=0,
                    max_seconds=30,unique_audio=True,exclude_manifests=[old_manifest],allow_stereo=True)
                same=prepare(root/'fresh.json',samples=6,revision='pinned',min_seconds=0,
                    max_seconds=30,unique_audio=True,exclude_manifests=[old_manifest],allow_stereo=True)
                with self.assertRaisesRegex(ValueError,'rejections:'):
                    prepare(root/'mono_only.json',samples=6,revision='pinned',min_seconds=0,
                        max_seconds=30,unique_audio=True,exclude_manifests=[old_manifest])
                diagnostic=json.loads((root/'mono_only.sampling_failure.json').read_text())
                self.assertEqual(diagnostic['accepted'],0)
                self.assertGreater(diagnostic['rejected']['speech_empty_or_stereo'],0)
            self.assertEqual(result,same)
            self.assertEqual(len({r['audio_sha256'] for r in result['rows']}),6)
            self.assertFalse({r['id'] for r in result['rows']}&{r['id'] for r in old})
            self.assertFalse({r['audio_sha256'] for r in result['rows']}&{r['audio_sha256'] for r in old})
            self.assertEqual(result['quotas'],dict(speech=2,sound=2,music=2))
            self.assertTrue(all(r['channels']==2 for r in result['rows']))
            self.assertEqual(result['settings']['audio_channel_policy'],'mono-or-stereo-mean-to-mono-v1')
            broken=deepcopy(result);broken['rows'][1]['audio_sha256']=broken['rows'][0]['audio_sha256']
            write_json(root/'broken.json',broken)
            with self.assertRaisesRegex(ValueError,'duplicate WAV'):read_manifest(root/'broken.json')

    def test_stereo_downmix_preserves_full_duration_and_mono_is_unchanged(self):
        from drift_study.data import load_audio
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'stereo.wav'
            samples=np.column_stack((np.linspace(-.8,.6,16000,dtype=np.float32),
                np.linspace(.4,.8,16000,dtype=np.float32)))
            sf.write(path,samples,16000,subtype='FLOAT')
            with self.assertRaisesRegex(ValueError,'explicitly enabled stereo'):load_audio(path)
            result=load_audio(path,allow_stereo=True)
            np.testing.assert_array_equal(result,samples.mean(axis=1))
            self.assertEqual(len(result),16000)
            sf.write(path,samples[:,0],16000,subtype='FLOAT')
            np.testing.assert_array_equal(load_audio(path),load_audio(path,allow_stereo=True))

    def test_200_preflight_accepts_matched_profiles_without_loading_gpu_or_models(self):
        from cache_study.run import main
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);audio=root/'clip.wav';audio.write_bytes(b'preflight-payload')
            rows=[dict(id=str(i),audio='clip.wav',audio_sha256=digest(audio),question='Q?',choices=['yes','no'],prompt=prompt_for('Q?',['yes','no'])) for i in range(200)]
            path=root/'cohort.json';write_json(path,dict(settings=dict(samples=200,input_protocol=INPUT_PROTOCOL),rows=rows))
            with patch.object(sys,'argv',['run','--manifest',str(path),'--samples','200','--matched-flat','--preflight-only']),\
                 patch('cache_study.run.verify_checkout') as checkout,patch('cache_study.run.load_model') as model,redirect_stdout(io.StringIO()) as output:
                main()
            checkout.assert_not_called();model.assert_not_called();self.assertIn('PREFLIGHT PASSED',output.getvalue())

    def test_actual200_launcher_freezes_both_cohorts_before_two_matched_jobs(self):
        import os,shlex,subprocess
        repository=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);scripts=root/'scripts';scripts.mkdir();binpath=root/'bin';binpath.mkdir()
            (scripts/'study200.sh').write_bytes((repository/'scripts/study200.sh').read_bytes())
            wrapper='#!'+sys.executable+'\n'+'''import sys,os,json,hashlib
from pathlib import Path
args=sys.argv[1:]
def value(flag):return args[args.index(flag)+1]
if args[:1]==['-u']:args=args[1:]
if args[:2]==['-m','drift_study.data']:
    from drift_study.protocol import prompt_for
    path=Path(value('--out'));path.parent.mkdir(parents=True,exist_ok=True)
    short='--max-seconds' in args;rows=[]
    for i in range(200):
        audio=path.parent/(path.stem+str(i)+'.wav');audio.write_bytes((path.stem+str(i)).encode())
        rows.append(dict(id=path.stem+str(i),audio=audio.name,audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
            audio_seconds=20 if short else120,question='Q?',choices=['yes','no'],prompt=prompt_for('Q?',['yes','no'])))
    path.write_text(json.dumps(dict(dataset='test',revision='pinned',settings=dict(samples=200,unique_audio=True,max_seconds=30 if short else None,audio_channel_policy='mono-or-stereo-mean-to-mono-v1'),rows=rows)))
    with open('calls.jsonl','a') as f:f.write(json.dumps(dict(kind='prepare',args=args))+'\\n')
elif args[:2]==['-m','cache_study.run']:
    with open('calls.jsonl','a') as f:f.write(json.dumps(dict(kind='preflight' if '--preflight-only' in args else 'decode',args=args))+'\\n')
    if '--out' in args:
        p=Path(value('--out'));p.mkdir(parents=True,exist_ok=True);(p/'stub.txt').write_text('job stub')
elif args[:2]==['-m','cache_study.paired']:
    p=Path(value('--out'));p.mkdir(parents=True,exist_ok=True);(p/'paired_statistics.html').write_text('statistics stub')
    with open('calls.jsonl','a') as f:f.write(json.dumps(dict(kind='statistics',args=args))+'\\n')
else:
    os.execv(sys.executable,[sys.executable]+args)
'''
            # A separated token is intentional Python syntax in the generated stub.
            wrapper=wrapper.replace('else120','else 120')
            (binpath/'python').write_text(wrapper);(binpath/'python').chmod(0o755)
            (scripts/'setup.sh').write_text('study_venv=/unused/venv\nstudy_diffa=/unused/official\nexport PATH='+shlex.quote(str(binpath))+':"$PATH"\n')
            from drift_study.data import DATASET_REVISION
            paths=[]
            for arm in ['old_short','old_long']:
                audio=root/(arm+'.wav');audio.write_bytes(arm.encode())
                rows=[dict(id=arm+str(i),audio=audio.name,audio_sha256=digest(audio),question='Q?',choices=['yes','no'],prompt=prompt_for('Q?',['yes','no'])) for i in range(50)]
                path=root/(arm+'.json');write_json(path,dict(dataset=DATASET,revision=DATASET_REVISION,settings=dict(samples=50),rows=rows));paths.append(path)
            env=dict(os.environ,ORIGINAL_MANIFEST=str(paths[0]),LONG_MANIFEST=str(paths[1]),PYTHONPATH=str(repository),
                SAMPLES='3',MAX_SECONDS='30',CACHE_START_FORWARD='57',DEEP_SEMANTIC_FACTOR='1',MATCHED_FLAT='0')
            for key in ['SHORT200_MANIFEST','LONG200_MANIFEST','SHORT200_RUN','LONG200_RUN','STATISTICS_RUN','RESULTS_ARCHIVE']:
                env.pop(key,None)
            done=subprocess.run(['bash','scripts/study200.sh'],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            calls=[json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual([c['kind'] for c in calls],['prepare','prepare','preflight','preflight','decode','decode','statistics'])
            for call in calls:
                args=call['args']
                if call['kind'] in ['prepare','preflight','decode']:
                    self.assertEqual(args[args.index('--samples')+1],'200')
                if call['kind']=='prepare':self.assertIn('--allow-stereo',args)
                if call['kind'] in ['preflight','decode']:
                    self.assertIn('--matched-flat',args)
                    self.assertEqual(args[args.index('--start-forward')+1],'10')
                    self.assertEqual(args[args.index('--deep-semantic-factor')+1],'2')
            self.assertIn('--max-seconds',calls[0]['args']);self.assertNotIn('--max-seconds',calls[1]['args'])
            plan=json.loads((root/'runs'/'mmau200_experimental_plan.json').read_text())
            self.assertEqual(plan['timed_generations'],5200);self.assertEqual(plan['conditions_per_question'],13)
            self.assertTrue((root/'runs'/'mmau200_matched_results.tar.gz').is_file())
            # A fresh Vast machine needs neither old manifest nor old WAV files.
            env.pop('ORIGINAL_MANIFEST');env.pop('LONG_MANIFEST')
            done=subprocess.run(['bash','scripts/study200.sh'],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            calls=[json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()][7:]
            self.assertFalse(any('--exclude-manifest' in c['args'] for c in calls))
            self.assertIn('previous-cohort overlap is unchecked',done.stdout)


class MatchedStatisticsTests(unittest.TestCase):
    @staticmethod
    def records(n):
        return {str(i):dict(id=str(i),audio_sha256='sha'+str(i),failed=False,
            score=dict(correct=i<n//2,target='A'),seconds=5.,output=dict(visible_tokens=100)) for i in range(n)}

    def test_200_question_effects_pairing_timing_and_exact_statistics(self):
        a=self.records(200);b=deepcopy(a)
        for i in range(95,100):b[str(i)]['score']['correct']=False
        for i in range(100,110):b[str(i)]['score']['correct']=True
        for r in b.values():r['seconds']=4.
        result=comparison(a,b,list(a),draws=300)
        self.assertEqual((result['gains'],result['losses'],result['accuracy_delta_pp']),(10,5,2.5))
        self.assertAlmostEqual(result['mcnemar_exact_p'],binomtest(10,15,.5).pvalue)
        self.assertAlmostEqual(result['clip_p'],result['mcnemar_exact_p'])
        self.assertAlmostEqual(result['paired_latency_speedup'],1.25)
        self.assertAlmostEqual(result['tps_gain_vs_reference_percent'],25.)
        bad=deepcopy(b);bad['0']['audio_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'mismatch'):comparison(a,bad,list(a),draws=100)
        failed=deepcopy(b);failed['0']['failed']=True
        result=comparison(a,failed,list(a),draws=100)
        self.assertEqual(result['losses'],6);self.assertEqual(result['paired_timing_n'],199)

    def test_cluster_dynamic_program_matches_exhaustive_swaps(self):
        deltas=[2,-1,0,-3,1]
        totals=[sum(s*d for s,d in zip(signs,deltas)) for signs in itertools.product((-1,1),repeat=len(deltas))]
        expected=sum(abs(v)>=abs(sum(deltas)) for v in totals)/len(totals)
        self.assertAlmostEqual(clip_signflip(deltas),expected)
        self.assertEqual(exact_mcnemar(0,0),1.)
        self.assertEqual(exact_mcnemar(4,0),.125)

    def test_complete_two_cohort_report_has_12_primary_24_secondary_and_rejects_missing_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);runs=[];policies={};schedules={}
            for name,base in POLICIES.items():
                policies[name]=base;schedules[name]=layer_schedule(base,list(range(32)))
                policies[name+'_deepsem2']=base;schedules[name+'_deepsem2']=layer_schedule(base,list(range(32)),list(range(24,32)),2)
            for cohort in ['short','long']:
                run=root/cohort;runs.append(run)
                config=dict(sample_ids=['0','1'],policies=policies,layer_intervals=schedules,manifest_sha256=cohort,
                    cache_layers=list(range(32)),deep_semantic_factor=2,deep_semantic_layers=list(range(24,32)),
                    calibration=dict(start_mode='fixed',dense_warmup_forwards=10),models={},
                    decoder=dict(max_answer_tokens=128,block_tokens=32,factor=1.,temperature=0.),seed=1,source_sha256='same')
                write_json(run/'cohort.json',dict(cohort=cohort,rows=[dict(id=i,audio_sha256=r['audio_sha256'],target='A') for i,r in self.records(2).items()]))
                config['manifest_sha256']=digest(run/'cohort.json')
                write_json(run/'config.json',config)
                for condition in ['baseline']+list(policies):
                    for i,r in self.records(2).items():write_json(run/'samples'/f'{int(i):05d}'/(condition+'.json'),r)
            result=analyze(runs,root/'stats',expected_samples=2,draws=100)
            self.assertEqual(sum(r['family']=='halved_vs_flat' for r in result['comparisons']),12)
            self.assertEqual(sum(r['family']=='versus_dense' for r in result['comparisons']),24)
            self.assertTrue(all(r['holm_family_p']==1. for r in result['comparisons']))
            self.assertTrue(all(r['accuracy_delta_ci95_pp'][0]<0<r['accuracy_delta_ci95_pp'][1] for r in result['comparisons']))
            self.assertTrue((root/'stats'/'paired_statistics.html').exists())
            (runs[0]/'samples'/'00001'/'888.json').unlink()
            with self.assertRaisesRegex(ValueError,'incomplete'):analyze(runs,root/'stats2',expected_samples=2,draws=100)


if __name__=='__main__':unittest.main()
