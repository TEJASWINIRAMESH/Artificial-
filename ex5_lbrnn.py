import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load and preprocess the text
filename = "wonderland.txt.txt"
with open(filename, 'r', encoding='utf-8') as file:
    raw_text = file.read().lower()

# Create a mapping of unique characters to integers
chars = sorted(list(set(raw_text)))
char_to_int = {c: i for i, c in enumerate(chars)}

# Summarize the data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters:", n_chars)
print("Total Vocab:", n_vocab)

# Prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(n_chars - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns:", n_patterns)

# Reshape and normalize input data, one-hot encode the output variable
X = np.reshape(dataX, (n_patterns, seq_length, 1)) / float(n_vocab)
y = to_categorical(dataY)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the LSTM model
model = Sequential([
    LSTM(256, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define a checkpoint callback
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=128, callbacks=callbacks_list)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Function to generate text
def generate_text(model, char_to_int, int_to_char, seed_text, length):
    result = []
    pattern = [char_to_int[char] for char in seed_text]
    
    for _ in range(length):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result.append(int_to_char[index])
        pattern.append(index)
        pattern = pattern[1:]
    
    return seed_text + ''.join(result)

# Map integers to characters
int_to_char = {i: c for i, c in enumerate(chars)}

# Generate and print text
seed_index = random.randint(0, n_chars - seq_length - 1)
seed_text = raw_text[seed_index:seed_index + seq_length]
generated_text = generate_text(model, char_to_int, int_to_char, seed_text, 1000)

print("Seed text:")
print(seed_text)
print("\nGenerated text:")
print(generated_text)
