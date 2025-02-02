# Assignment: Building a Text Generation Model with Python

# Step 1: Data Collection and Preprocessing
import os
import re
import requests
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('stopwords')

# Download text data from Project Gutenberg (example: "Pride and Prejudice")
def fetch_text():
    source_url = "https://www.gutenberg.org/files/1342/1342-0.txt"
    response = requests.get(source_url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Failed to fetch data from the source")

raw_text = fetch_text()

# Clean and prepare the text data
def clean_text(raw_data):
    # Remove unnecessary content from the dataset
    raw_data = re.split(r'\*\*\* START OF (.*?) \*\*\*', raw_data)[-1]
    raw_data = re.split(r'\*\*\* END OF (.*?) \*\*\*', raw_data)[0]

    # Tokenize sentences and remove unwanted characters
    sentences = sent_tokenize(raw_data)
    processed_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence).lower()
        processed_sentences.append(sentence)

    return processed_sentences

processed_text = clean_text(raw_text)

# Save cleaned data to a file
with open("processed_text_data.txt", "w") as text_file:
    for line in processed_text:
        text_file.write(line + "\n")

# Step 2: Model Creation and Training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Tokenize the cleaned text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_text)
vocab_size = len(tokenizer.word_index) + 1

# Create input-output sequences
sequence_data = []
for sentence in processed_text:
    tokens = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokens)):
        ngram_sequence = tokens[:i + 1]
        sequence_data.append(ngram_sequence)

# Pad sequences to have consistent lengths
max_seq_len = max([len(seq) for seq in sequence_data])
sequence_data = pad_sequences(sequence_data, maxlen=max_seq_len, padding='pre')

X_train = sequence_data[:, :-1]
y_train = sequence_data[:, -1]
y_train = tf.keras.utils.to_categorical(y_train, num_classes=vocab_size)

# Build the text generation model
text_model = Sequential([
    Embedding(vocab_size, 100, input_length=max_seq_len - 1),
    LSTM(150, return_sequences=True),
    LSTM(100),
    Dense(vocab_size, activation='softmax')
])

text_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
NUM_EPOCHS = 20
text_model.fit(X_train, y_train, epochs=NUM_EPOCHS, verbose=1)

# Save the trained model
text_model.save("text_gen_model.h5")

# Step 3: Generating Text
# Function to generate sentences using the trained model
def generate_sentence(seed_text, trained_model, tokenizer_obj, gen_length=50):
    for _ in range(gen_length):
        token_sequence = tokenizer_obj.texts_to_sequences([seed_text])[0]
        token_sequence = pad_sequences([token_sequence], maxlen=max_seq_len - 1, padding='pre')
        predicted_probs = trained_model.predict(token_sequence, verbose=0)
        predicted_word_index = np.argmax(predicted_probs)
        next_word = tokenizer_obj.index_word.get(predicted_word_index, '')
        if next_word == '':
            break
        seed_text += ' ' + next_word
    return seed_text

# Test the text generation
starting_text = "It is a truth universally acknowledged"
generated_sentence = generate_sentence(starting_text, text_model, tokenizer)
print("Generated Text:\n", generated_sentence)
