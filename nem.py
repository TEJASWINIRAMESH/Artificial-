# Install necessary libraries
!pip install transformers
!pip install tensorflow
!pip install tensorflow-datasets

import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, create_optimizer
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Load the CoNLL-2003 dataset
data, info = tfds.load('conll2003', with_info=True)

# Set up the tokenizer and model
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=info.features['ner_tags'].feature.num_classes)

# Function to tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, return_tensors="tf", padding="max_length", max_length=128)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        tokenized_inputs["labels"] = tf.convert_to_tensor(label_ids)
    return tokenized_inputs

# Tokenize the datasets
train_dataset = data['train'].map(tokenize_and_align_labels, batched=True)
validation_dataset = data['validation'].map(tokenize_and_align_labels, batched=True)
test_dataset = data['test'].map(tokenize_and_align_labels, batched=True)

# Prepare the TensorFlow datasets
train_dataset = train_dataset.to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'labels'],
    label_cols=['labels'],
    shuffle=True,
    batch_size=8,
    collate_fn=lambda x: x
)

validation_dataset = validation_dataset.to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'labels'],
    label_cols=['labels'],
    shuffle=False,
    batch_size=8,
    collate_fn=lambda x: x
)

# Compile and train the model
num_train_steps = len(train_dataset)
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
)

loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
model.compile(optimizer=optimizer, loss=loss)

model.fit(train_dataset, validation_data=validation_dataset, epochs=3)

# Evaluate the model
results = model.evaluate(test_dataset)
print(f"Test loss: {results}")

# Making predictions
example = next(iter(test_dataset))
inputs = {key: value for key, value in example.items() if key != 'labels'}
outputs = model(inputs)
predictions = tf.argmax(outputs.logits, axis=-1)
print(predictions)