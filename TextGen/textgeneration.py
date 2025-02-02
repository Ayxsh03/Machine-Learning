import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("TextGen/text_gen_model.h5")

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.word_index = {"example": 1, "words": 2, "used": 3, "in": 4, "training": 5}

# Function to generate text
def generate_text(model, tokenizer, seed_text, max_sequence_len, num_words_to_generate):
    for _ in range(num_words_to_generate):
        tokenized_input = tokenizer.texts_to_sequences([seed_text])[0]
        tokenized_input = pad_sequences([tokenized_input], maxlen=max_sequence_len - 1, padding='pre')
        predicted_token = np.argmax(model.predict(tokenized_input), axis=-1)[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break
        if output_word == "":
            break
        seed_text += " " + output_word

    return seed_text

seed_text = "The journey of a thousand miles"
max_sequence_len = 50  # Same as used during training
num_words_to_generate = 20

# Generate text
generated_text = generate_text(model, tokenizer, seed_text, max_sequence_len, num_words_to_generate)
print("Generated Text:")
print(generated_text)