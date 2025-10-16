import requests
samples = [
    'good morning bro',
    'i hate this so much',
    'this is the worst movie ever',
    'i love this',
    'terrible, do not buy'
]
for s in samples:
    r = requests.post('http://127.0.0.1:5000/predict', json={'text': s})
    try:
        print(s, '->', r.status_code, r.json())
    except Exception:
        print(s, '->', r.status_code, r.text)
