from flask import Flask, request, jsonify, render_template_string
import numpy as np
# TensorFlow/Keras imports are heavy; import them lazily inside functions (get_model/predict)
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / 'sentiment_model.h5'
WORD_INDEX_PATH = Path(__file__).parent / 'word_index.json'
MAXLEN = 200
TOP_WORDS = 10000
THRESH_PATH = Path(__file__).parent / 'threshold_eval.json'

# Simple HTML chat UI (Bootstrap bubble layout)
HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sentiment Chat</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { padding: 1.5rem; background: #f8fafc; }
    .chat-container { display:flex; gap:1rem; }
    .chat-box { flex:1; display:flex; flex-direction:column; }
    .messages { flex:1; min-height:300px; max-height:60vh; overflow:auto; padding:1rem; background:#ffffff; border-radius:8px; border:1px solid #e6eef6; }
    .input-row { margin-top:0.75rem; display:flex; gap:0.5rem; }
    .msg-bubble { padding:0.6rem 0.9rem; border-radius:12px; display:inline-block; max-width:80%; }
    .from-user { background:#e6ffed; align-self:flex-end; border-bottom-right-radius:4px; }
    .from-bot { background:#eef2ff; align-self:flex-start; border-bottom-left-radius:4px; }
    .meta { font-size:0.75rem; color:#6b7280; margin-top:0.25rem; }
    .sentiment-badge { font-weight:600; padding:0.25rem 0.5rem; border-radius:6px; }
    .confidence { height:8px; background:#e6e7ee; border-radius:4px; overflow:hidden; margin-top:6px; }
    .confidence > div { height:100%; background:#34d399; }
  </style>
</head>
<body>
  <div class="container">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <h1 class="h4">Sentiment Chat</h1>
      <div>
        <a class="btn btn-outline-secondary me-2" href="/dashboard">Dashboard</a>
        <button id="clearBtn" class="btn btn-sm btn-outline-danger">Clear</button>
      </div>
    </div>

    <div class="chat-container">
      <div class="chat-box">
        <div id="messages" class="messages" aria-live="polite"></div>
        <div class="input-row">
          <input id="t" class="form-control" placeholder="Type a message and press Enter" aria-label="Message input" />
          <button id="sendBtn" class="btn btn-primary">Send</button>
        </div>
      </div>
      <div style="width:320px">
        <div class="card mb-3">
          <div class="card-body">
            <h5 class="card-title">Model Info</h5>
            <p class="mb-1"><code>sentiment_model.h5</code></p>
            <p class="text-muted small">Max tokens: <strong>200</strong></p>
          </div>
        </div>
        <div class="card">
          <div class="card-body">
            <h6>Usage</h6>
            <p class="small">Enter text and hit Enter or the Send button. Predictions are logged and visible on the dashboard.</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const messages = document.getElementById('messages');
    const input = document.getElementById('t');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');

    function escapeText(s){ const t = document.createTextNode(s); const span = document.createElement('span'); span.appendChild(t); return span; }

    function addMessage(who, text, isUser=false, meta=null){
      const wrap = document.createElement('div'); wrap.style.display='flex'; wrap.style.flexDirection='column'; wrap.style.marginBottom='0.6rem';
      const bubble = document.createElement('div'); bubble.className='msg-bubble ' + (isUser? 'from-user':'from-bot');
      bubble.appendChild(escapeText(text));
      wrap.appendChild(bubble);
      if(meta){
        const metaEl = document.createElement('div'); metaEl.className='meta';
        if(meta.time) metaEl.appendChild(escapeText(meta.time + (meta.source? ' • ' + meta.source : '')));
        if(meta.sentiment){
          const badge = document.createElement('span'); badge.className='sentiment-badge ms-2'; badge.style.marginLeft='8px';
          badge.textContent = meta.sentiment; badge.style.background = meta.sentiment === 'Positive' ? '#bbf7d0' : '#fecaca'; badge.style.color='#111827';
          metaEl.appendChild(badge);
        }
        if(typeof meta.score === 'number'){
          const conf = document.createElement('div'); conf.className='confidence'; const inner = document.createElement('div'); inner.style.width = Math.round(meta.score*100) + '%';
          conf.appendChild(inner); metaEl.appendChild(conf);
        }
        wrap.appendChild(metaEl);
      }
      messages.appendChild(wrap);
      messages.scrollTop = messages.scrollHeight;
      return wrap;
    }

    async function sendMessage(){
      const text = input.value.trim();
      if(!text) return;
      // create and keep reference to user's message wrapper so we can attach prediction metadata to it
      const userWrap = addMessage('You', text, true, {time: new Date().toLocaleTimeString()});
      input.value = '';
      sendBtn.disabled = true; input.disabled = true;
      // add placeholder bot message
      const placeholder = addMessage('Bot', 'Thinking...', false, {time: new Date().toLocaleTimeString()});
      try{
        const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
          if(!res.ok){
            const err = await res.text(); placeholder.querySelector('.msg-bubble').textContent = 'Error: ' + (err || res.statusText || res.status);
          } else {
            const j = await res.json();
            const score = (typeof j.score === 'number') ? j.score : parseFloat(j.score) || 0;
            // replace placeholder with a neutral acknowledgement (sentiment is attached to user's bubble only)
            placeholder.querySelector('.msg-bubble').textContent = 'Prediction ready';
          // attach prediction metadata to user's message bubble (only)
          try{
            const existingUserMeta = userWrap.querySelector('.meta'); if(existingUserMeta) existingUserMeta.remove();
            const userMeta = document.createElement('div'); userMeta.className='meta'; userMeta.appendChild(escapeText(new Date().toLocaleTimeString()));

            // helper: interpolate from red -> green based on score (0..1)
            function scoreToColor(s){
              const r1=220, g1=38, b1=38; // red
              const r2=22, g2=163, b2=74; // green
              const r = Math.round(r1 + (r2 - r1) * s);
              const g = Math.round(g1 + (g2 - g1) * s);
              const b = Math.round(b1 + (b2 - b1) * s);
              return `rgb(${r},${g},${b})`;
            }

            const displayCategory = j.category || j.sentiment || '—';
            // user requested to show 'Slightly Positive' as Negative
            let displayLabel = displayCategory;
            if(displayCategory === 'Slightly Positive') displayLabel = 'Negative';

            const userBadge = document.createElement('span'); userBadge.className='sentiment-badge ms-2';
            userBadge.textContent = displayLabel;
            const badgeColor = j.color || scoreToColor(score);
            userBadge.style.background = badgeColor;
            userBadge.style.color = '#111827'; userBadge.style.marginLeft = '8px';
            userMeta.appendChild(userBadge);
            // numeric score shown next to badge for clarity
            try{
              const scoreEl = document.createElement('span'); scoreEl.style.fontWeight = 600; scoreEl.style.marginLeft = '8px'; scoreEl.textContent = (score*100).toFixed(0) + '%';
              userMeta.appendChild(scoreEl);
              // show star rating if provided
              if(j.rating){
                const stars = '⭐'.repeat(Math.max(1, Math.min(5, j.rating)));
                const rEl = document.createElement('span'); rEl.style.marginLeft = '8px'; rEl.textContent = stars + ' ' + j.rating + '/5';
                userMeta.appendChild(rEl);
              }
            }catch(e){ }

            const userConf = document.createElement('div'); userConf.className='confidence';
            const innerU = document.createElement('div'); innerU.style.width = Math.round(score*100) + '%';
            innerU.style.background = scoreToColor(score);
            userConf.appendChild(innerU); userMeta.appendChild(userConf);
            userWrap.appendChild(userMeta);
          } catch(e){ /* non-critical UI attach failed */ }
        }
      } catch(err){
        placeholder.querySelector('.msg-bubble').textContent = 'Error: ' + (err.message || err);
      } finally{
        sendBtn.disabled = false; input.disabled = false; input.focus();
      }
    }

    // event listeners
    sendBtn.addEventListener('click', (e)=>{ e.preventDefault(); sendMessage(); });
    input.addEventListener('keydown', (e)=>{ if(e.key === 'Enter'){ e.preventDefault(); sendMessage(); } });
    clearBtn.addEventListener('click', ()=>{ messages.innerHTML = ''; });
  </script>
</body>
</html>
'''

# Load model and word index lazily
_model = None
_word_index = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError('Model file not found. Run hhe.py to train and save the model as sentiment_model.h5')
    # lazy import to avoid importing TensorFlow at module import time
    try:
      from tensorflow.keras.models import load_model
    except Exception:
      from keras.models import load_model
    _model = load_model(str(MODEL_PATH))
    return _model


def get_threshold():
  # if a computed threshold exists from evaluation, use it; otherwise default 0.5
  if THRESH_PATH.exists():
    try:
      j = json.loads(THRESH_PATH.read_text(encoding='utf-8'))
      th = float(j.get('best_threshold') or j.get('best_th') or j.get('threshold') or 0.5)
      return th
    except Exception:
      return 0.5
  return 0.5


def get_word_index():
  """Return the raw IMDB word_index (word -> original index)
  We'll add the +3 offset when converting tokens to sequences so behavior matches imdb.load_data.
  """
  global _word_index
  if _word_index is None:
    if not WORD_INDEX_PATH.exists():
      raise RuntimeError('word_index.json not found. Run hhe.py to generate it')
    raw = json.loads(WORD_INDEX_PATH.read_text())
    # ensure ints
    _word_index = {k: int(v) for k, v in raw.items()}
  return _word_index


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/dashboard')
def dashboard():
    # lightweight dashboard page that fetches history and predictions
    dash_html = '''
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Sentiment Dashboard</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-light">
      <div class="container py-4">
        <div class="d-flex justify-content-between align-items-center mb-4">
          <h1 class="h3">Sentiment Model Dashboard</h1>
          <a class="btn btn-outline-secondary" href="/">Open Chat</a>
        </div>

        <div class="row g-3 mb-4">
          <div class="col-md-4">
            <div class="card p-3">
              <h6 class="mb-2">Model</h6>
              <p class="mb-1"><code>sentiment_model.h5</code></p>
              <p class="text-muted small">Max tokens: <strong>200</strong></p>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card p-3">
              <h6 class="mb-2">Last Train</h6>
              <p id="lastTrain">—</p>
            </div>
          </div>
          <div class="col-md-4">
            <div class="card p-3">
              <h6 class="mb-2">Recent Predictions</h6>
              <p id="predCount">—</p>
            </div>
          </div>
        </div>

        <div class="row mb-4">
          <div class="col-md-6">
            <div class="card p-3">
              <canvas id="accChart"></canvas>
            </div>
          </div>
          <div class="col-md-6">
            <div class="card p-3">
              <canvas id="lossChart"></canvas>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-body">
            <h5 class="card-title">Recent Predictions</h5>
            <div class="table-responsive">
              <table class="table table-sm table-striped" id="predTable"><thead><tr><th>Time</th><th>Text</th><th>Sentiment</th><th>Score</th></tr></thead><tbody></tbody></table>
            </div>
          </div>
        </div>

      </div>

      <script>
        async function load() {
          const h = await fetch('/api/history').then(r=>r.json()).catch(()=>null);
          if (h && h.history) {
            const hist = h.history;
            const labels = (hist.loss||[]).map((_,i)=>i+1);
            const acc = hist.accuracy || hist.acc || [];
            const val_acc = hist.val_accuracy || hist.val_acc || [];
            const loss = hist.loss || [];
            const val_loss = hist.val_loss || [];
            document.getElementById('lastTrain').textContent = h.saved_at || 'unknown';
            new Chart(document.getElementById('accChart').getContext('2d'), {type:'line',data:{labels, datasets:[{label:'Train Acc',data:acc,borderColor:'#16a34a',backgroundColor:'rgba(16,163,74,0.05)',fill:true},{label:'Val Acc',data:val_acc,borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,0.05)',fill:true}]}, options:{responsive:true}});
            new Chart(document.getElementById('lossChart').getContext('2d'), {type:'line',data:{labels, datasets:[{label:'Train Loss',data:loss,borderColor:'#dc2626',backgroundColor:'rgba(220,38,38,0.05)',fill:true},{label:'Val Loss',data:val_loss,borderColor:'#f97316',backgroundColor:'rgba(249,115,22,0.05)',fill:true}]}, options:{responsive:true}});
          }
          const p = await fetch('/api/predictions').then(r=>r.json()).catch(()=>[]);
          document.getElementById('predCount').textContent = p.length;
          const tbody = document.querySelector('#predTable tbody');
          tbody.innerHTML = '';
          p.slice(-200).reverse().forEach(e=>{
            const tr = document.createElement('tr');
            const txt = e.text.length>120? e.text.slice(0,120)+'...': e.text;
            tr.innerHTML = `<td>${e.time}</td><td>${escapeHtml(txt)}</td><td>${e.sentiment}</td><td>${(e.score).toFixed(3)}</td>`;
            tbody.appendChild(tr);
          });
        }
        function escapeHtml(unsafe) { return unsafe.replace(/[&<>"']/g, function(m){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m];}); }
        load();
      </script>
    </body>
    </html>
    '''
    return render_template_string(dash_html)


@app.route('/api/history')
def api_history():
    p = Path(__file__).parent / 'history.json'
    if not p.exists():
        return jsonify({})
    return jsonify(json.loads(p.read_text()))


@app.route('/api/predictions')
def api_predictions():
    p = Path(__file__).parent / 'predictions.log'
    if not p.exists():
        return jsonify([])
    lines = [l.strip() for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    out = []
    for ln in lines[-200:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return jsonify(out)


@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json()
  text = (data.get('text') or '').strip()
  if not text:
    return jsonify({'error': 'empty text'}), 400

  # ensure word_index is loaded
  try:
    word_index = get_word_index()
  except Exception as e:
    return jsonify({'error': str(e)}), 500

  # tokenization (prefer Keras helper, fallback to simple regex)
  try:
    from tensorflow.keras.preprocessing.text import text_to_word_sequence
  except Exception:
    try:
      from keras.preprocessing.text import text_to_word_sequence
    except Exception:
      import re
      def text_to_word_sequence(s):
        return re.findall(r"[A-Za-z0-9']+", s.lower())

  tokens = text_to_word_sequence(text)

  # Map tokens to imdb indices used in training: mapped = raw_index + 3
  seq = []
  for w in tokens:
    raw_idx = word_index.get(w)
    if raw_idx is None:
      seq.append(2)  # UNK
      continue
    mapped = int(raw_idx) + 3
    if mapped >= TOP_WORDS:
      seq.append(2)
    else:
      seq.append(mapped)

  # pad and predict
  try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
  except Exception:
    from keras.preprocessing.sequence import pad_sequences
  padded = pad_sequences([seq], maxlen=MAXLEN)
  model = get_model()
  score = float(model.predict(padded)[0][0])

  # debug output (printed to server console)
  try:
    print('DEBUG predict:', {'text': text, 'tokens': tokens[:20], 'seq_sample': seq[:20], 'score': score})
  except Exception:
    pass

  # decide binary sentiment using evaluated threshold (better than fixed 0.5)
  threshold = get_threshold()
  sentiment = 'Positive' if score >= threshold else 'Negative'

  # detailed multi-level category mapping (for richer color-coded UI)
  # scale: Very Negative < Negative < Slightly Negative < Slightly Positive < Positive < Very Positive
  if score >= 0.85:
    category = 'Very Positive'
    color = '#16a34a'
  elif score >= 0.65:
    category = 'Positive'
    color = '#34d399'
  elif score >= threshold:
    category = 'Slightly Positive'
    color = '#bbf7d0'
  elif score >= max(0.0, threshold - 0.10):
    category = 'Slightly Negative'
    color = '#fecaca'
  elif score >= 0.35:
    category = 'Negative'
    color = '#f87171'
  else:
    category = 'Very Negative'
    color = '#dc2626'

  # Server-side rule: if message contains strong negative words, force Negative
  NEGATIVE_KEYWORDS = {'hate','terrible','worst','awful','bad','boring','disappoint','dislike','sucks','horrible','trash','stupid','worse','dont',"don't",'no','not'}
  text_lower = text.lower()
  has_negative = any(w in tokens or w in text_lower for w in NEGATIVE_KEYWORDS)
  if has_negative:
    # force negative prediction for clear negative language
    sentiment = 'Negative'
    category = 'Very Negative'
    color = '#dc2626'
    # optionally dampen the score so UI reflects negative
    try:
      score = float(min(score, 0.2))
    except Exception:
      pass

  # Server-side rule: treat 'Slightly Positive' as Negative (per user request)
  if category == 'Slightly Positive':
    sentiment = 'Negative'
    category = 'Negative'
    color = '#f87171'

  # server-side logging of predictions for dashboard
  try:
    logp = Path(__file__).parent / 'predictions.log'
    # map score (0..1) to a 1-5 product rating (simple linear mapping)
    rating = int(round(score * 4.0)) + 1
    if rating < 1: rating = 1
    if rating > 5: rating = 5

    entry = {'time': datetime.utcnow().isoformat(), 'text': text, 'score': score, 'sentiment': sentiment, 'category': category, 'rating': rating}
    with logp.open('a', encoding='utf-8') as f:
      f.write(json.dumps(entry) + '\n')
  except Exception:
    pass

  return jsonify({'sentiment': sentiment, 'score': score, 'category': category, 'color': color, 'rating': rating})


if __name__ == '__main__':
    app.run(debug=True)
