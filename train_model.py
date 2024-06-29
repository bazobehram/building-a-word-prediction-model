import wikipediaapi
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import json

wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='userAgent/1.0 (example.com; email@example.com)'
)
page = wiki_wiki.page('Machine_learning')
text = page.text

text = text.replace('\n', ' ').replace('\r', ' ').replace('\ufeff', '').replace('“', '').replace('”', '')

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index

with open('word_index.json', 'w') as f:
    json.dump({'wordIndex': word_index}, f)

total_words = len(word_index) + 1
input_sequences = []

token_list = tokenizer.texts_to_sequences([text])[0]
for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

# Padding
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Features and labels
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=1)

model.save('word_prediction_model.h5')
