# %%
import keras
from keras import ops # operations for tensor manipulation functions
from keras import layers
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# %%
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# %%
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# %%
vocab_size = 20000 # only top 20K words
max_len = 200 # only top 200 words in each movie review

(X_train, y_train), (X_val, y_val) = imdb.load_data(num_words=vocab_size)

# %%
X_train = pad_sequences(X_train, maxlen=max_len)
X_val = pad_sequences(X_val, maxlen=max_len)

# %%
inputs = layers.Input(shape=(max_len,))

# %%
embed_dim = 32
embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
x = embedding_layer(inputs)

# %%
num_heads = 2 # attention heads
ff_dim = 32 # hidden layer size in FFN inside transformer
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)

# %%
x = layers.GlobalAveragePooling1D()(x)

# %%
x = layers.Dropout(0.1)(x)

# %%
x = layers.Dense(20, activation='relu')(x)

# %%
x = layers.Dropout(0.1)(x)

# %%
outputs = layers.Dense(2, activation='softmax')(x)

# %%
model = keras.Model(inputs=inputs, outputs=outputs)

# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_val, y_val))

# %%
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(imdb.get_word_index().keys())


def predict_sentiment(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded_sequence = pad_sequences(sequence, maxlen=200)

    prediction = model.predict(padded_sequence)
    sentiment = np.argmax(prediction, axis=1)[0]
    sentiment_label = "positive" if sentiment == 0 else "negative"
    return sentiment_label


user_input = "This movie was bad.Its is boring"
sentiment = predict_sentiment(user_input)
print(f"The sentiment of the input text is: {sentiment}")



