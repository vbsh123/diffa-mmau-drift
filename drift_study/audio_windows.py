"""Complete-clip windowing and projected-token budgeting for the pinned encoder."""
import math
import struct
from types import SimpleNamespace

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 30*SAMPLE_RATE  # Encoder window, never a clip-duration filter.
ACOUSTIC_PER_WINDOW = 64
INPUT_PROTOCOL = 'complete-clip-chronological-window-tokens-v2'


def window_ranges(samples, window_samples=WINDOW_SAMPLES):
    if samples <= 0 or window_samples <= 0:
        raise ValueError('Empty audio or invalid encoder window')
    count = (samples+window_samples-1)//window_samples
    # Balanced contiguous windows avoid tiny tails; every sample occurs exactly once.
    return [(i*samples//count,(i+1)*samples//count) for i in range(count)]


def projected_tokens(frames, sample_rate):
    if frames <= 0 or sample_rate <= 0:
        raise ValueError('Invalid audio header')
    resampled = (frames*SAMPLE_RATE+sample_rate-1)//sample_rate
    ranges = window_ranges(resampled)
    # Feature extractor mask: every160 samples; Whisper conv stride2; projector /4.
    semantic = sum((((end-start+159)//160+1)//2)//4 for start,end in ranges)
    acoustic = len(ranges)*ACOUSTIC_PER_WINDOW
    return dict(resampled_samples=resampled,windows=len(ranges),semantic=semantic,
                acoustic=acoustic,total=semantic+acoustic)


def wav_header(stream):
    """Read RIFF/RF64 PCM/float WAV sizes without fetching the entire waveform."""
    header = stream.read(12)
    if len(header)!=12 or header[:4] not in (b'RIFF',b'RF64') or header[8:]!=b'WAVE':
        raise ValueError('Not a RIFF/RF64 WAV')
    fmt, data_size64 = None,None
    while True:
        chunk = stream.read(8)
        if len(chunk)!=8:
            raise ValueError('Incomplete WAV header')
        name,size = struct.unpack('<4sI',chunk)
        if name==b'data':
            if fmt is None:
                raise ValueError('WAV data before format')
            channels,rate,align = fmt
            size = data_size64 if size==0xffffffff else size
            if size is None or not align or not channels or not rate:
                raise ValueError('Invalid WAV sizes')
            return SimpleNamespace(frames=size//align,samplerate=rate,channels=channels)
        if name==b'fmt ':
            content = stream.read(size)
            if len(content)!=size or size<16:
                raise ValueError('Invalid WAV format')
            code,channels,rate,_,align,_ = struct.unpack('<HHIIHH',content[:16])
            if code not in (1,3,0xfffe):
                raise ValueError('Unsupported compressed WAV header')
            fmt=channels,rate,align
        elif name==b'ds64':
            content=stream.read(size)
            if len(content)<24:
                raise ValueError('Invalid RF64 sizes')
            _,data_size64,_=struct.unpack('<QQQ',content[:24])
        else:
            # Skip metadata incrementally; no payload-size admission filter.
            remaining=size
            while remaining:
                block=stream.read(min(remaining,65536))
                if not block:
                    raise ValueError('Incomplete WAV metadata')
                remaining-=len(block)
        if size%2:
            stream.read(1)
