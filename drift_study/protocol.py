"""Study settings, input-position groups, and final-answer scoring."""
from dataclasses import dataclass
import re

GROUPS = ('semantic', 'acoustic', 'text')
DOMAINS = ('speech', 'sound', 'music')
SEED = 20261025
ANSWER_TOKENS = 128
BLOCK_TOKENS = 32
FACTOR = 1.0
PAIRWISE_LAYERS = (0, 7, 15, 23, 31)
INSTRUCTION = (
    'Answer the following question using the audio. Think step by step and give a brief '
    'explanation grounded in what you hear. Then end with a separate line in exactly '
    'this format: Final answer: <letter>. Choose one of the listed options.\n\n'
)


def prompt_for(question, choices):
    if not 2 <= len(choices) <= 26:
        raise ValueError('Expected 2–26 MCQ options')
    return INSTRUCTION + question.strip() + '\n' + '\n'.join(
        f'{chr(65+i)}. {choice}' for i, choice in enumerate(choices))


def grade(text, target):
    markers = list(re.finditer(r'final\s+answer\s*:', text, re.I))
    tail = text[markers[-1].end():].strip() if markers else ''
    match = re.fullmatch(r'\(?([A-Z])\)?[.!]?', tail, re.I)
    prediction = match.group(1).upper() if match else None
    return dict(prediction=prediction, target=target, correct=prediction == target,
                parse_failed=prediction is None, scorer='explicit-final-letter-v1')


@dataclass
class Prepared:
    prompt: object
    speech: object
    insertion: int
    acoustic_count: int
    mask_id: int
    audio_chunks: object = None

    @property
    def prefix_length(self):
        return int(self.prompt.shape[1] + self.speech.shape[1])

    def groups(self):
        audio_end = self.insertion + int(self.speech.shape[1])
        if not 0 < self.acoustic_count < int(self.speech.shape[1]):
            raise ValueError('Expected nonempty acoustic and semantic groups')
        if self.audio_chunks is None:
            acoustic=list(range(self.insertion,self.insertion+self.acoustic_count))
            semantic=list(range(self.insertion+self.acoustic_count,audio_end))
        else:
            acoustic,semantic,position=[],[],self.insertion
            for chunk in self.audio_chunks:
                a,s=chunk['acoustic_tokens'],chunk['semantic_tokens']
                acoustic.extend(range(position,position+a))
                semantic.extend(range(position+a,position+a+s))
                position+=a+s
            if position!=audio_end or len(acoustic)!=self.acoustic_count:
                raise ValueError('Audio window layout does not match projected tokens')
        return dict(acoustic=acoustic,semantic=semantic,
                    text=list(range(self.insertion))+list(range(audio_end,self.prefix_length)))
