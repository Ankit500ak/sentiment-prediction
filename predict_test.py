import json
from pathlib import Path
import re

p = Path(__file__).parent
model_path = p / 'sentiment_model.h5'
word_index_path = p / 'word_index.json'

if not model_path.exists() or not word_index_path.exists():
    print('Missing model or word_index.json')
    raise SystemExit(2)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model(str(model_path))
with word_index_path.open('r', encoding='utf-8') as f:
    word_index = json.load(f)

TOP_WORDS = 10000
MAXLEN = 300

def toks(s):
    return re.findall(r"[A-Za-z0-9']+", s.lower())

def text_to_seq(text):
    t = toks(text)
    seq = []
    for w in t:
        raw = word_index.get(w)
        if raw is None:
            seq.append(2)
            continue
        mapped = int(raw) + 3
        if mapped >= TOP_WORDS:
            seq.append(2)
        else:
            seq.append(mapped)
    return seq

samples = [
    'good morning bro',
    'I hate this so much',
    'this is the worst movie ever',
    'i love this',
    'terrible, do not buy',
    'amazing work, well done'
]

for s in samples:
    seq = text_to_seq(s)
    padded = pad_sequences([seq], maxlen=MAXLEN)
    score = float(model.predict(padded)[0][0])
    print(f"{s!r} -> {score:.4f} -> {'Positive' if score>0.5 else 'Negative'}")
