import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

root = Path(__file__).resolve().parent.parent
data_folder = root / "data"
pcl_path = data_folder / "dontpatronizeme.tsv"
pcl_labels_path = data_folder / "dontpatronizeme_labels.tsv"
train_labels_path = data_folder / "train_labels.csv"
dev_labels_path = data_folder / "dev_labels.csv"
test_path = data_folder / "test.tsv"

PCL_COLUMNS = ["par_id", "art_id", "keyword", "country_code", "text", "label"]

def download_pcl_data():
    data_folder.mkdir(parents=True, exist_ok=True)
    if not pcl_path.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/CRLala/NLPLabs-2024/refs/heads/main/Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv",
            pcl_path
        )
    if not pcl_labels_path.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/CRLala/NLPLabs-2024/refs/heads/main/Dont_Patronize_Me_Trainingset/dontpatronizeme_categories.tsv",
            pcl_labels_path
        )
    if not test_path.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/refs/heads/master/semeval-2022/TEST/task4_test.tsv",
            test_path
        )

def download_train_splits():
    data_folder.mkdir(parents=True, exist_ok=True)
    if not train_labels_path.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/refs/heads/master/semeval-2022/practice%20splits/train_semeval_parids-labels.csv",
            train_labels_path
        )
    if not dev_labels_path.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/refs/heads/master/semeval-2022/practice%20splits/dev_semeval_parids-labels.csv",
            dev_labels_path
        )

def load_pcl_tsv_to_df(tsv_path: str | Path, skiprows: int = 4) -> pd.DataFrame:
    p = Path(tsv_path)
    return pd.read_csv(
        p, sep="\t", header=None, names=PCL_COLUMNS, skiprows=skiprows,
        dtype={"par_id": "int64", "art_id": "string", "keyword": "string", "country_code": "string", "text": "string", "label": "int64"},
        keep_default_na=False,
    )

def add_binary_label(df: pd.DataFrame, graded_col: str = "label", out_col: str = "binary_label", threshold: int = 2) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = (out[graded_col].astype(int) >= threshold).astype(int)
    return out

def load_split_ids(train_labels_csv: str | Path = train_labels_path, dev_labels_csv: str | Path = dev_labels_path, id_col: str = "par_id") -> tuple[set[int], set[int]]:
    train_ids = set(pd.read_csv(train_labels_csv)[id_col].astype(int).tolist())
    dev_ids = set(pd.read_csv(dev_labels_csv)[id_col].astype(int).tolist())
    return train_ids, dev_ids

def make_splits(df: pd.DataFrame, train_ids: set[int], dev_ids: set[int], id_col: str = "par_id") -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    dev_df = df[df[id_col].isin(dev_ids)].reset_index(drop=True)
    return train_df, dev_df

def create_internal_splits(train_df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    internal_train, internal_dev = train_test_split(
        train_df, test_size=test_size, random_state=seed, stratify=train_df["binary_label"]
    )
    return internal_train.reset_index(drop=True), internal_dev.reset_index(drop=True)

# --- EDA Functions ---
def label_distribution_table(train_df: pd.DataFrame, dev_df: pd.DataFrame, label_col: str = "binary_label") -> pd.DataFrame:
    train_counts = train_df[label_col].value_counts().sort_index()
    dev_counts = dev_df[label_col].value_counts().sort_index()
    out = pd.DataFrame({"train_count": train_counts, "dev_count": dev_counts}).fillna(0)
    out["train_prop"] = (out["train_count"] / out["train_count"].sum()).round(4)
    out["dev_prop"] = (out["dev_count"] / out["dev_count"].sum()).round(4)
    return out

def plot_label_distribution(train_df: pd.DataFrame, dev_df: pd.DataFrame, label_col: str = "binary_label"):
    train_counts = train_df[label_col].value_counts().sort_index()
    dev_counts = dev_df[label_col].value_counts().sort_index()
    plot_df = pd.DataFrame({
        "split": ["train", "train", "dev", "dev"],
        "label": [0, 1, 0, 1],
        "count": [int(train_counts.get(0, 0)), int(train_counts.get(1, 0)), int(dev_counts.get(0, 0)), int(dev_counts.get(1, 0))]
    })
    plt.figure(figsize=(6, 4))
    sns.barplot(data=plot_df, x="label", y="count", hue="split")
    plt.xticks([0, 1], ["No PCL (0)", "PCL (1)"])
    plt.title("Binary label distribution (train vs dev)")
    plt.tight_layout()
    plt.show()

def add_word_length(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    out = df.copy()
    out["word_len"] = out[text_col].astype(str).apply(lambda s: len(s.split()))
    return out

def length_percentiles_by_class(df: pd.DataFrame, length_col: str = "word_len", label_col: str = "binary_label", percentiles: tuple[int, ...] = (10, 25, 50, 75, 90, 95, 99)) -> pd.DataFrame:
    rows = []
    for lbl, sub in df.groupby(label_col):
        arr = sub[length_col].to_numpy()
        row = {"class": "No PCL (0)" if int(lbl) == 0 else "PCL (1)", "n": int(len(sub)), "mean": float(arr.mean()) if len(arr) else 0.0, "std": float(arr.std(ddof=0)) if len(arr) else 0.0, "max": int(arr.max()) if len(arr) else 0}
        if len(arr):
            vals = np.percentile(arr, percentiles)
            for p, v in zip(percentiles, vals):
                row[f"p{p}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)

def clip_max_mean_plus_k_std(df: pd.DataFrame, length_col: str = "word_len", k: float = 4.0) -> int:
    x = df[length_col].astype(float).to_numpy()
    return max(1, int(np.ceil(float(np.mean(x)) + k * float(np.std(x, ddof=0))))) if len(x) else 1

def plot_length_histogram_by_class_probability(df: pd.DataFrame, length_col: str = "word_len", label_col: str = "binary_label", bins: int = 60, clip_k: float = 4.0, clip_max: Optional[int] = None):
    if clip_max is None: clip_max = clip_max_mean_plus_k_std(df, length_col=length_col, k=clip_k)
    x0, x1 = df.loc[df[label_col] == 0, length_col].to_numpy(), df.loc[df[label_col] == 1, length_col].to_numpy()
    bin_edges = np.linspace(0, clip_max, bins + 1)
    w0, w1 = (np.ones_like(x0, dtype=float) / len(x0)) if len(x0) else None, (np.ones_like(x1, dtype=float) / len(x1)) if len(x1) else None
    plt.figure(figsize=(6, 4))
    plt.hist(x0, bins=bin_edges, weights=w0, alpha=0.6, label="No PCL (0)")
    plt.hist(x1, bins=bin_edges, weights=w1, alpha=0.6, label="PCL (1)")
    plt.xlabel("Word count"); plt.ylabel("Probability"); plt.title(f"Text length distribution by class (probability, clipped at mean + {clip_k}·std = {clip_max})")
    plt.legend(); plt.tight_layout(); plt.show()

def proportion_above_clip_by_class(df: pd.DataFrame, clip_max: int, length_col: str = "word_len", label_col: str = "binary_label") -> pd.DataFrame:
    rows = []
    for lbl in [0, 1]:
        x = df[df[label_col] == lbl][length_col].to_numpy()
        rows.append({"class": "No PCL (0)" if lbl == 0 else "PCL (1)", "clip_max": int(clip_max), "prop_above_clip": round(float((x > clip_max).mean()), 4) if len(x) else 0.0})
    return pd.DataFrame(rows)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Tokenizes the texts and formats them into a PyTorch Dataset.
        """
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=False,
            max_length=max_length
        )

        if isinstance(labels, pd.Series):
            self.labels = labels.tolist()
        else:
            self.labels = list(labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

