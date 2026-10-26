"""Exercise the actual launcher with cheap job stubs and hostile inherited settings."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from drift_study.protocol import prompt_for
from drift_study.run import verify_cache_reference
from drift_study.storage import digest,write_json


class ExperimentAuditTests(unittest.TestCase):
    def test_measurement_reference_rejects_changed_output_work_or_inputs(self):
        base=dict(id='q',audio_sha256='wav',forward_count=21,output=dict(canvas_ids=[1,2]),
                  group_tokens=dict(acoustic=64,semantic=500,text=100),audio_chunks=[dict(start_sample=0,end_sample=100)])
        reference=dict(base,failed=False)
        verify_cache_reference(base,reference)
        mutations=[('forward_count',22),('audio_sha256','other'),('id','other'),
                   ('output',dict(canvas_ids=[1,3])),('group_tokens',dict(acoustic=64,semantic=499,text=100)),
                   ('audio_chunks',[]),('failed',True)]
        for key,value in mutations:
            changed=deepcopy(reference);changed[key]=value
            with self.subTest(field=key),self.assertRaisesRegex(ValueError,'clean cache baseline'):
                verify_cache_reference(base,changed)

    def test_actual_three_job_launcher_locks_protocol_and_packages_outputs(self):
        repository=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);scripts=root/'scripts';scripts.mkdir()
            (scripts/'next.sh').write_bytes((repository/'scripts'/'next.sh').read_bytes())
            (scripts/'setup.sh').write_text(
                'study_venv='+shlex.quote(str(Path(sys.executable).parent.parent))+'\n'
                'study_diffa=/unused/official\n'
                'export PATH='+shlex.quote(str(Path(sys.executable).parent))+':"$PATH"\n')
            stub='''import json,os
from pathlib import Path
keys=('MANIFEST','RUN_NAME','SAMPLES','COHORT_SAMPLES','MIN_SECONDS','MAX_SECONDS','CACHE_LAYERS',
      'CACHE_START','CACHE_START_FORWARD','DEEP_SEMANTIC_FACTOR','DEEP_SEMANTIC_LAYERS',
      'MATCHED_FLAT','ALLOW_LEGACY_SHORT_COHORT','ATTENTION_EVERY','ATTENTION_QUERY_CHUNK',
      'PAIRWISE_DOMAINS','VERIFY_SAMPLES','CACHE_REFERENCE_RUN','BASELINE_RUN')
row={key:os.environ.get(key) for key in keys}
Path(row['RUN_NAME']).mkdir(parents=True)
with open('jobs.jsonl','a') as stream: stream.write(json.dumps(row)+'\\n')
'''
            for name in ('cache.sh','run.sh'):
                (scripts/name).write_text(shlex.quote(sys.executable)+" - <<'PY'\n"+stub+'PY\n')
            audio=root/'clip.wav';audio.write_bytes(b'unchanged-test-payload')
            original=root/'original.json'
            rows=[dict(id=str(i),audio='clip.wav',audio_sha256=digest(audio),
                       question='Q?',choices=['yes','no'],prompt=prompt_for('Q?',['yes','no'])) for i in range(50)]
            write_json(original,dict(settings=dict(samples=50),rows=rows))
            env=dict(os.environ,ORIGINAL_MANIFEST=str(original),SAMPLES='3',COHORT_SAMPLES='3',
                     MAX_SECONDS='30',MIN_SECONDS='0',ATTENTION_EVERY='0',CACHE_LAYERS='0',
                     MATCHED_FLAT='1',BASELINE_RUN='/bad/reference',PAIRWISE_DOMAINS='0',VERIFY_SAMPLES='0',
                     PYTHONPATH=str(repository))
            for key in ('LONG_MANIFEST','ORIGINAL_CACHE_RUN_NAME','LONG_CACHE_RUN_NAME','DRIFT_RUN_NAME','RESULTS_ARCHIVE'):
                env.pop(key,None)
            completed=subprocess.run(['bash','scripts/next.sh'],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
            jobs=[json.loads(line) for line in (root/'jobs.jsonl').read_text().splitlines()]
            self.assertEqual(len(jobs),3)
            self.assertEqual(jobs[0]['MANIFEST'],str(original))
            self.assertEqual(jobs[1]['MANIFEST'],jobs[2]['MANIFEST'])
            self.assertEqual(jobs[2]['CACHE_REFERENCE_RUN'],jobs[1]['RUN_NAME'])
            self.assertEqual([j['ALLOW_LEGACY_SHORT_COHORT'] for j in jobs[:2]],['1','0'])
            for job in jobs:
                for key,value in dict(SAMPLES='50',COHORT_SAMPLES='50',MIN_SECONDS='60',CACHE_LAYERS='all',
                                      MATCHED_FLAT='0',ATTENTION_EVERY='1',VERIFY_SAMPLES='2',PAIRWISE_DOMAINS='3').items():
                    self.assertEqual(job[key],value)
                self.assertIsNone(job['MAX_SECONDS'])
                self.assertIsNone(job['BASELINE_RUN'])
            for job in jobs[:2]:
                self.assertEqual((job['CACHE_START'],job['CACHE_START_FORWARD'],job['DEEP_SEMANTIC_FACTOR'],job['DEEP_SEMANTIC_LAYERS']),
                                 ('fixed','10','2','24-31'))
            import tarfile
            with tarfile.open(root/'runs'/'next_results.tar.gz') as archive:
                self.assertEqual(set(archive.getnames()),{j['RUN_NAME'].split('/')[-1] for j in jobs})

    def test_unhalved_launcher_uses_existing_cohorts_and_only_two_cache_jobs(self):
        repository=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);scripts=root/'scripts';scripts.mkdir()
            (scripts/'controls.sh').write_bytes((repository/'scripts'/'controls.sh').read_bytes())
            (scripts/'setup.sh').write_text(
                'study_venv='+shlex.quote(str(Path(sys.executable).parent.parent))+'\n'
                'study_diffa=/unused/official\n'
                'export PATH='+shlex.quote(str(Path(sys.executable).parent))+':"$PATH"\n')
            stub="""import json,os
from pathlib import Path
row={key:os.environ.get(key) for key in ('MANIFEST','RUN_NAME','SAMPLES','CACHE_START',
    'CACHE_START_FORWARD','DEEP_SEMANTIC_FACTOR','CACHE_LAYERS','MATCHED_FLAT',
    'ALLOW_LEGACY_SHORT_COHORT','MAX_SECONDS','BASELINE_RUN','VERIFY_SAMPLES')}
Path(row['RUN_NAME']).mkdir(parents=True)
with open('jobs.jsonl','a') as stream:stream.write(json.dumps(row)+'\\n')
"""
            (scripts/'cache.sh').write_text(shlex.quote(sys.executable)+" - <<'PY'\n"+stub+'PY\n')
            audio=root/'clip.wav';audio.write_bytes(b'frozen-payload')
            rows=[dict(id=str(i),audio='clip.wav',audio_sha256=digest(audio),
                question='Q?',choices=['yes','no'],prompt=prompt_for('Q?',['yes','no'])) for i in range(50)]
            original=root/'original.json';long=root/'long.json'
            write_json(original,dict(settings=dict(samples=50),rows=rows))
            from drift_study.audio_windows import INPUT_PROTOCOL
            write_json(long,dict(settings=dict(samples=50,input_protocol=INPUT_PROTOCOL,min_seconds=60,max_seconds=None),rows=rows))
            before=[digest(p) for p in (original,long)]
            env=dict(os.environ,ORIGINAL_MANIFEST=str(original),LONG_MANIFEST=str(long),
                PYTHONPATH=str(repository),SAMPLES='3',CACHE_START='mean',CACHE_START_FORWARD='57',
                DEEP_SEMANTIC_FACTOR='2',MAX_SECONDS='30',MATCHED_FLAT='1',CACHE_LAYERS='0',
                BASELINE_RUN='/bad/reference',VERIFY_SAMPLES='0')
            for key in ('ORIGINAL_CACHE_RUN_NAME','LONG_CACHE_RUN_NAME','RESULTS_ARCHIVE'):
                env.pop(key,None)
            done=subprocess.run(['bash','scripts/controls.sh'],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            jobs=[json.loads(line) for line in (root/'jobs.jsonl').read_text().splitlines()]
            self.assertEqual(len(jobs),2)
            self.assertEqual([j['MANIFEST'] for j in jobs],[str(original),str(long)])
            self.assertEqual([digest(p) for p in (original,long)],before)
            self.assertEqual([j['ALLOW_LEGACY_SHORT_COHORT'] for j in jobs],['1','0'])
            for job in jobs:
                for key,value in dict(SAMPLES='50',CACHE_START='fixed',CACHE_START_FORWARD='10',
                    DEEP_SEMANTIC_FACTOR='1',CACHE_LAYERS='all',MATCHED_FLAT='0',VERIFY_SAMPLES='2').items():
                    self.assertEqual(job[key],value)
                self.assertIsNone(job['MAX_SECONDS']);self.assertIsNone(job['BASELINE_RUN'])
            self.assertTrue((root/'runs'/'forward10_unhalved_controls.tar.gz').is_file())


if __name__=='__main__':
    unittest.main()
