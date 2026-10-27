"""Seeded, duration-verified MMAU-Pro sampling without whole-archive download."""
import argparse
import hashlib
from collections import Counter
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import zipfile

from .protocol import ANSWER_TOKENS, DOMAINS, SEED, prompt_for
from .remote_zip import HTTPRangeFile
from .storage import digest, write_json
from .audio_windows import INPUT_PROTOCOL, projected_tokens, wav_header

DATASET = 'gamma-lab-umd/MMAU-Pro'
DATASET_REVISION = 'b8516c217464d53be36d6dc334fda3cd578ed103'


def quotas(count):
    return {domain: count//3+int(i<count%3) for i, domain in enumerate(DOMAINS)}


def metadata_eligible(row):
    choices = row.get('choices') or []
    return (bool(re.fullmatch(r'[A-Za-z0-9_-]{1,100}', str(row.get('id', ''))))
            and row.get('category') in DOMAINS
            and len(row.get('audio_path') or []) == 1
            and bool(row.get('question')) and 2 <= len(choices) <= 26
            and len(set(choices)) == len(choices) and choices.count(row.get('answer')) == 1)


def read_manifest(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    rows = document['rows']
    if len(rows) != document['settings']['samples'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Manifest count or IDs invalid')
    if document['settings'].get('unique_audio') and len({r['audio_sha256'] for r in rows})!=len(rows):
        raise ValueError('Unique-audio manifest contains duplicate WAV hashes')
    for row in rows:
        audio = path.parent/row['audio']
        if not audio.is_file() or digest(audio) != row['audio_sha256']:
            raise ValueError(f'Audio hash mismatch: {audio}')
        if row['prompt'] != prompt_for(row['question'], row['choices']):
            raise ValueError('Manifest prompt differs from current protocol')
    return document


def context_spec():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from .model import LLM_REPO,LLM_REVISION
    path=hf_hub_download(LLM_REPO,'config.json',revision=LLM_REVISION)
    config=json.loads(Path(path).read_text())
    return AutoTokenizer.from_pretrained(LLM_REPO,revision=LLM_REVISION),int(config['max_sequence_length'])


def prompt_token_count(tokenizer,instruction):
    text=('<|startoftext|><|start_header_id|>system<|end_header_id|>\n\n'
          'You are a helpful voice assistant.<|eot_id|>'
          '<|start_header_id|>user<|end_header_id|>\n\nThis is the audio: <audio>. '
          +instruction+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    return len(tokenizer(text)['input_ids'])


def prepare(out,samples=50,seed=SEED,revision=DATASET_REVISION,max_seconds=None,min_seconds=60.,
            unique_audio=False,exclude_manifests=()):
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    import pyarrow.parquet as pq
    import soundfile as sf
    out = Path(out).resolve()
    settings = dict(samples=samples,seed=seed,max_seconds=max_seconds,min_seconds=min_seconds,
                    input_protocol=INPUT_PROTOCOL)
    exclusions=[read_manifest(path) for path in exclude_manifests]
    excluded_ids={r['id'] for document in exclusions for r in document['rows']}
    excluded_hashes={r['audio_sha256'] for document in exclusions for r in document['rows']}
    excluded_members={r['source_audio'] for document in exclusions for r in document['rows'] if r.get('source_audio')}
    if unique_audio:
        settings['unique_audio']=True
    if exclude_manifests:
        settings['exclude_manifest_sha256']=sorted(digest(path) for path in exclude_manifests)
    if any(document['dataset']!=DATASET or document['revision']!=revision for document in exclusions):
        raise ValueError('Exclusion manifests must use the same pinned MMAU-Pro revision')
    if (samples<3 or not math.isfinite(min_seconds) or min_seconds<0
            or (max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds<=min_seconds))):
        raise ValueError('Need >=3 examples and valid optional duration bounds; no default maximum')
    if out.exists():
        existing = read_manifest(out)
        if existing['dataset']!=DATASET or existing['settings'] != settings or revision not in ('main', existing['revision']):
            raise ValueError('Preparation settings changed; use a new manifest')
        return existing
    pinned = HfApi().dataset_info(DATASET, revision=revision).sha
    tokenizer,context_limit=context_spec()
    metadata = hf_hub_download(DATASET, 'test.parquet', repo_type='dataset', revision=pinned)
    rows = pq.read_table(metadata).to_pylist()
    pools = {domain: sorted([r for r in rows if metadata_eligible(r) and r['category'] == domain], key=lambda r:r['id'])
             for domain in DOMAINS}
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = out.parent/(out.stem+'_audio')
    audio_dir.mkdir(exist_ok=True)
    rejected, accepted = Counter(), {domain: [] for domain in DOMAINS}
    remote = HTTPRangeFile(hf_hub_url(DATASET, 'data.zip', repo_type='dataset', revision=pinned))
    seen_audio=set()
    seen_members=set()
    with remote, zipfile.ZipFile(remote) as archive:
        names = set(archive.namelist())
        for domain, needed in quotas(samples).items():
            for candidate in pools[domain]:
                member = candidate['audio_path'][0]
                if candidate['id'] in excluded_ids or member in excluded_members:
                    rejected[domain+'_previous_question_or_audio']+=1
                    continue
                if unique_audio and member in seen_members:
                    rejected[domain+'_duplicate_audio_member']+=1
                    continue
                if PurePosixPath(member).is_absolute() or '..' in PurePosixPath(member).parts:
                    raise ValueError('Unsafe archive path')
                # The published metadata uses data/..., while archive roots can differ.
                matches = [n for n in names if n == member or n.endswith('/'+member)]
                if not matches:
                    matches = [n for n in names if PurePosixPath(n).name == PurePosixPath(member).name]
                if len(matches) != 1:
                    raise ValueError(f'Cannot uniquely locate {member} in archive')
                info = archive.getinfo(matches[0])
                # Inspect full duration from RIFF sizes before fetching large candidates.
                payload=None
                try:
                    with archive.open(info) as stream:
                        description=wav_header(stream)
                except ValueError:
                    payload=archive.read(info)
                    description=sf.info(io.BytesIO(payload))
                duration = description.frames/description.samplerate
                if duration<=min_seconds or (max_seconds is not None and duration>max_seconds):
                    rejected[domain+'_duration'] += 1
                    continue
                if not description.frames or description.channels != 1:
                    rejected[domain+'_empty_or_stereo'] += 1
                    continue
                instruction=prompt_for(candidate['question'],candidate['choices'])
                text_tokens=prompt_token_count(tokenizer,instruction)
                estimate=projected_tokens(description.frames,description.samplerate)
                if estimate['total']+text_tokens+ANSWER_TOKENS>context_limit:
                    rejected[domain+'_context_budget']+=1
                    continue
                if payload is None:
                    payload=archive.read(info)
                actual=sf.info(io.BytesIO(payload))
                if (actual.frames,actual.samplerate,actual.channels)!=(description.frames,description.samplerate,description.channels):
                    raise ValueError('Full audio header differs from streamed header; refusing truncated or inconsistent audio')
                audio_hash=hashlib.sha256(payload).hexdigest()
                if audio_hash in excluded_hashes:
                    rejected[domain+'_previous_audio_hash']+=1
                    continue
                if unique_audio and audio_hash in seen_audio:
                    rejected[domain+'_duplicate_audio_hash']+=1
                    continue
                seen_audio.add(audio_hash)
                seen_members.add(member)
                target = chr(65+candidate['choices'].index(candidate['answer']))
                destination = audio_dir/(candidate['id']+'.wav')
                destination.write_bytes(payload)
                accepted[domain].append(dict(id=candidate['id'], domain=domain,
                    question=candidate['question'], choices=candidate['choices'], target=target,
                    prompt=instruction,estimated_audio_tokens=estimate,text_tokens=text_tokens,
                    audio=str(destination.relative_to(out.parent)), audio_sha256=digest(destination),
                    audio_seconds=duration, sample_rate=description.samplerate, channels=description.channels,
                    source_audio=member, length_type=candidate.get('length_type'),
                    perceptual_skills=candidate.get('perceptual_skills'), reasoning_skills=candidate.get('reasoning_skills')))
                print(f'Prepared {domain} {len(accepted[domain])}/{needed}: {duration:.2f}s; '
                      f'{estimate["windows"]} windows, {estimate["total"]} audio tokens',flush=True)
                if len(accepted[domain]) == needed:
                    break
            if len(accepted[domain]) != needed:
                raise ValueError(f'Only {len(accepted[domain])}/{needed} eligible {domain} examples; no fallback sampling')
    # Interleave domains: the first three are predetermined matrix exemplars.
    selected = [accepted[d][i] for i in range(max(map(len, accepted.values()))) for d in DOMAINS if i<len(accepted[d])]
    document = dict(dataset=DATASET, revision=pinned, metadata_sha256=digest(metadata), settings=settings,
                    quotas=quotas(samples), exclusions=dict(rejected), archive_bytes_read=remote.bytes_received,
                    metadata_rows=len(rows), metadata_eligible_rows=sum(map(len, pools.values())), rows=selected)
    document['context_limit']=context_limit
    document['unique_audio_files']=len({r['audio_sha256'] for r in selected})
    write_json(out, document)
    return document


def load_audio(path):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if audio.shape[1] != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError('Need complete finite mono audio')
    wave = torch.from_numpy(audio[:, 0].copy())
    if rate != 16000:
        wave = torchaudio.functional.resample(wave, rate, 16000)
    return wave.numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('data/mmau50_long.json'))
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--revision', default=DATASET_REVISION)
    p.add_argument('--min-seconds',type=float,default=60.)
    p.add_argument('--max-seconds',type=float,default=None)
    p.add_argument('--unique-audio',action='store_true')
    p.add_argument('--exclude-manifest',type=Path,action='append',default=[])
    args = p.parse_args()
    prepare(args.out,args.samples,args.seed,args.revision,args.max_seconds,args.min_seconds,
            args.unique_audio,args.exclude_manifest)


if __name__ == '__main__':
    main()
