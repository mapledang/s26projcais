import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datasets import Dataset
from sklearn.metrics import classification_report
from preprocess import load_and_preprocess

MODEL_PATH = "./model"

def main():
    _, _, test_df = load_and_preprocess()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

    test_ds = Dataset.from_pandas(test_df)

    def tokenize(batch):
        return tokenizer(
            batch["body"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    test_ds = test_ds.map(tokenize, batched=True)
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    preds = []
    labels = []

    model.eval()
    for batch in test_ds:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attention_mask"].unsqueeze(0)
            )
        preds.append(outputs.logits.argmax(dim=1).item())
        labels.append(batch["label"].item())

    print(classification_report(labels, preds, target_names=["Legitimate", "Phishing"]))

if __name__ == "__main__":
    main()
