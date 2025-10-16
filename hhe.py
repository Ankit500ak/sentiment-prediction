# Sentiment Analysis using LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from tensorflow.keras.datasets import imdb
except Exception:
    from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json
from pathlib import Path
from datetime import datetime

# Load IMDB dataset
num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences to ensure equal length
maxlen = 200
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Build model
model = Sequential([
    Embedding(num_words, 128, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test))

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Save model, word index and training history for serving/dashboard
MODEL_PATH = Path(__file__).parent / 'sentiment_model.h5'
WORD_INDEX_PATH = Path(__file__).parent / 'word_index.json'
HISTORY_PATH = Path(__file__).parent / 'history.json'

model.save(str(MODEL_PATH))
try:
    word_index = imdb.get_word_index()
    WORD_INDEX_PATH.write_text(json.dumps(word_index))
    # save history (convert numpy floats to Python floats)
    h = {k: [float(x) for x in v] for k, v in history.history.items()}
    meta = {'saved_at': datetime.utcnow().isoformat(), 'history': h}
    HISTORY_PATH.write_text(json.dumps(meta))
    print(f"Saved model to {MODEL_PATH}, word index to {WORD_INDEX_PATH}, history to {HISTORY_PATH}")
except Exception as e:
    print('Could not save word index or history:', e)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Predict function
def predict_sentiment(text, model, word_index):
    from tensorflow.keras.preprocessing.text import text_to_word_sequence
    tokens = text_to_word_sequence(text)
    seq = [word_index.get(word, 0) for word in tokens]
    padded = pad_sequences([seq], maxlen=maxlen)
    pred = model.predict(padded)[0][0]
    sentiment = "Positive 😀" if pred > 0.5 else "Negative 😞"
    print(f"Text: {text}\nPrediction: {sentiment}")

# Example test
word_index = imdb.get_word_index()
predict_sentiment("This movie was absolutely fantastic!", model, word_index)
predict_sentiment("The film was boring and disappointing.", model, word_index)