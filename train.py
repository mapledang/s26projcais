import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from preprocess import load_and_preprocess

MODEL_NAME = "distilbert-base-uncased"

def tokenize(batch, tokenizer):
    return tokenizer(
        batch["body"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

def main():
    train_df, val_df, _ = load_and_preprocess()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model("./model")

if __name__ == "__main__":
    main()
