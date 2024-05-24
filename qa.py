# %%
!pip3 install git+https://github.com/huggingface/transformers.git
!pip3 install datasets
!pip3 install huggingface-hub

# %%
from datasets import load_dataset

datasets = load_dataset("squad")

# %%
print(datasets["train"][0])

# %%
from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# %%
max_length = 384  # The maximum length of a feature (question and context)
doc_stride = (
    128  # The authorized overlap between two part of the context when splitting
)

# %%
def prepare_train_features(examples):
    from transformers import AutoTokenizer

    model_checkpoint = "distilbert-base-cased"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    max_length = 384  # The maximum length of a feature (question and context)
    doc_stride = (
    128  # The authorized overlap between two part of the context when splitting
)
    examples["question"] = [q.lstrip() for q in examples["question"]]
    examples["context"] = [c.lstrip() for c in examples["context"]]
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )


    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:

                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

# %%


tokenized_datasets = datasets.map(
    prepare_train_features,
    batched=True,
    remove_columns=datasets["train"].column_names,
    num_proc=3,
)


# %%
train_set = tokenized_datasets["train"].with_format("numpy")[
    :
]  # Load the whole dataset as a dict of numpy arrays
validation_set = tokenized_datasets["validation"].with_format("numpy")[:]

# %%
from transformers import TFAutoModelForQuestionAnswering

model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# %%
import tensorflow as tf
from tensorflow import keras

optimizer = keras.optimizers.Adam(learning_rate=5e-5)

# %%
# Optionally uncomment the next line for float16 training
keras.mixed_precision.set_global_policy("mixed_float16")

model.compile(optimizer=optimizer)

# %%
model.fit(train_set, validation_data=validation_set, epochs=1)

# %%
context = """Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. It also has extensive documentation and developer guides. """
question = "What is Keras?"

inputs = tokenizer([context], [question], return_tensors="np")
outputs = model(inputs)
start_position = tf.argmax(outputs.start_logits, axis=1)
end_position = tf.argmax(outputs.end_logits, axis=1)
print(int(start_position), int(end_position[0]))

# %%
answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
print(answer)

# %%
print(tokenizer.decode(answer))



