# %%
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN,Dense

# %%
vocab_size = 10000  # Number of words to consider as features
maxlen = 300        # Cut texts after this number of words
embedding_dim = 128

# %%
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# %%
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# %%
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen))
model.add(SimpleRNN(units=64))  # SimpleRNN layer with 64 units
model.add(Dense(units=1, activation='sigmoid'))

# %%
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# %%

history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))

# %%
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# %%
def preprocess_input(text, tokenizer, maxlen):
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequences
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

# %%
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

# Instantiate a tokenizer and fit on the training data
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts([reverse_word_index.get(i - 3, "?") for i in range(3, vocab_size + 3)])

# %%
# Custom input text
custom_text = "I did not enjoy the movie. It was boring and too long.The music is also bad"

# Preprocess the custom input
custom_input = preprocess_input(custom_text, tokenizer, maxlen)

# Predict the sentiment of the custom input
prediction = model.predict(custom_input)
print(f"Predicted Sentiment: {'Positive' if prediction[0][0] > 0.7 else 'Negative'}")


