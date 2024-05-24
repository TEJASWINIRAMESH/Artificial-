# %%
!pip install transformers datasets evaluate

# %%
from datasets import load_dataset
eli5 = load_dataset("eli5_category", split="train[:5000]")

# %%
eli5 = eli5.train_test_split(test_size=0.2)

# %%
eli5["train"][0]

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

# %%
eli5 = eli5.flatten()
eli5["train"][0]

# %%
def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

# %%
tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)

# %%
block_size = 128


def group_texts(examples):

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

# %%
lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

# %%
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")

# %%
from transformers import create_optimizer, AdamWeightDecay

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

# %%
from transformers import TFAutoModelForMaskedLM

model = TFAutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")

# %%
tf_train_set = model.prepare_tf_dataset(
    lm_dataset["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    lm_dataset["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

# %%
import tensorflow as tf

model.compile(optimizer=optimizer)  # No loss argument!

# %%
model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3)

# %%
text = "The Milky Way is a <mask> galaxy."

# %%
from transformers import pipeline

mask_filler = pipeline("fill-mask", "username/my_awesome_eli5_mlm_model")
mask_filler(text, top_k=3)

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_mlm_model")
inputs = tokenizer(text, return_tensors="tf")
mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]

# %%
from transformers import TFAutoModelForMaskedLM

model = TFAutoModelForMaskedLM.from_pretrained("username/my_awesome_eli5_mlm_model")
logits = model(**inputs).logits
mask_token_logits = logits[0, mask_token_index, :]

# %%
top_3_tokens = tf.math.top_k(mask_token_logits, 3).indices.numpy()

for token in top_3_tokens:
    print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))


