import pandas as pd
import re
from sklearn.model_selection import train_test_split

DATA_PATH = "data/phishing_email.csv"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"\S+@\S+", "EMAIL", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    df = df[["body", "label"]].dropna()

    df["body"] = df["body"].apply(clean_text)
    df["label"] = df["label"].map({"legitimate": 0, "spam": 1})

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    return train_df, val_df, test_df
