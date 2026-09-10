import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.special import softmax

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import src.data_utils as du

def write_predictions_to_file(predictions, filepath):
    with open(filepath, 'w') as f:
        for pred in predictions:
            f.write(f"{int(pred)}\n")
    print(f"Wrote {len(predictions)} predictions to {filepath}")

def main():
    model_dir = PROJECT_ROOT / "BestModel" / "final_model"
    output_dir = PROJECT_ROOT
    os.makedirs(output_dir, exist_ok=True)

    # Extract the classification threshold
    csv_path = PROJECT_ROOT / "BestModel" / "grid_results.csv"
    grid_df = pd.read_csv(csv_path)
    best_threshold = float(grid_df.loc[grid_df['best_f1'].idxmax()]['best_threshold'])
    print(f"Using threshold: {best_threshold:.4f}")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=32,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator
    )

    print("Loading datasets")
    du.download_pcl_data()
    df = du.load_pcl_tsv_to_df(du.pcl_path)
    df = du.add_binary_label(df, threshold=2)
    train_ids, dev_ids = du.load_split_ids()
    _, official_dev_df = du.make_splits(df, train_ids, dev_ids)

    test_df = pd.read_csv(
        du.test_path,
        sep='\t',
        header=None,
        names=['par_id', 'art_id', 'keyword', 'country', 'text']
    )

    test_df['text'] = test_df['text'].fillna("").astype(str)

    print(f"Loaded Test Set: {len(test_df)} rows")

    dummy_test_labels = np.zeros(len(test_df))

    dev_ds = du.TextDataset(official_dev_df["text"], np.zeros(len(official_dev_df)), tokenizer)
    test_ds = du.TextDataset(test_df['text'].tolist(), dummy_test_labels, tokenizer)

    # Generate Predictions for dev
    print("\nPredicting on official dev")
    dev_preds_output = trainer.predict(dev_ds)
    dev_probs = softmax(dev_preds_output.predictions, axis=1)[:, 1]
    dev_binary_preds = (dev_probs >= best_threshold).astype(int)

    dev_output_file = output_dir / "dev.txt"
    write_predictions_to_file(dev_binary_preds, dev_output_file)

    # Generate Predictions for test
    print("\nPredicting on official test")
    test_preds_output = trainer.predict(test_ds)
    test_probs = softmax(test_preds_output.predictions, axis=1)[:, 1]
    test_binary_preds = (test_probs >= best_threshold).astype(int)

    test_output_file = output_dir / "test.txt"
    write_predictions_to_file(test_binary_preds, test_output_file)

if __name__ == "__main__":
    main()