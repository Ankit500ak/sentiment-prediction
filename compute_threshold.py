import json
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

p = Path(__file__).parent
model_path = p / 'sentiment_model.h5'
if not model_path.exists():
    print('Model missing')
    raise SystemExit(2)

TOP_WORDS = 10000
MAXLEN = 300

model = load_model(str(model_path))

print('Loading IMDB test set (this may download if not present)')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=TOP_WORDS)
print('Loaded', len(x_test), 'test samples')

x_test_p = pad_sequences(x_test, maxlen=MAXLEN)
probs = model.predict(x_test_p, batch_size=256, verbose=1).ravel()
auc = roc_auc_score(y_test, probs)
print('ROC AUC:', auc)

# search thresholds for best F1
ths = np.linspace(0.01, 0.99, 99)
best = {'th':0.5, 'f1':0}
for th in ths:
    preds = (probs >= th).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best['f1']:
        best = {'th':th, 'f1':f1}

print('Best threshold by F1:', best)
# show precision/recall at default 0.5 and best
for th in (0.5, best['th']):
    preds = (probs >= th).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_test, preds, average='binary')
    print(f"th={th:.2f} -> precision={p:.3f} recall={r:.3f} f1={f:.3f}")

# Save recommended threshold
out = {'roc_auc': float(auc), 'best_threshold': float(best['th']), 'best_f1': float(best['f1'])}
out_path = Path(__file__).parent / 'threshold_eval.json'
with out_path.open('w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print('Saved', out_path)
