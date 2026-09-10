from pathlib import Path
import sys, os, random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import src.data_utils as du
import src.augment as aug
import src.trainer as tr


def main():
    final_output_dir = PROJECT_ROOT / "BestModel" / "final_model"
    os.makedirs(final_output_dir, exist_ok=True)

    csv_path = PROJECT_ROOT / "BestModel" / "grid_results.csv"
    grid_df = pd.read_csv(csv_path)

    # Get best model from grid search
    best_run = grid_df.loc[grid_df['best_f1'].idxmax()]
    best_alpha = float(best_run['alpha'])
    best_lr = float(best_run['lr'])
    best_pos_weight = float(best_run['pos_weight'])
    best_ratio = float(best_run['ratio'])
    best_threshold = float(best_run['best_threshold'])

    print("Best Hyperparameters")
    print(f"Alpha (Augmentation Noise) : {best_alpha}")
    print(f"Negative-to-Positive Ratio : {best_ratio}")
    print(f"Learning Rate              : {best_lr}")
    print(f"Positive Class Weight      : {best_pos_weight}")
    print(f"Validation F1 (Internal)   : {best_run['best_f1']:.4f}")
    print(f"Internal validation thres  : {best_threshold:.4f}")

    # Load dataset
    df = du.load_pcl_tsv_to_df(du.pcl_path)
    df = df.dropna(subset=['text', 'label'])
    df = du.add_binary_label(df, threshold=2)

    train_ids, dev_ids = du.load_split_ids()
    official_train_df, official_dev_df = du.make_splits(df, train_ids, dev_ids)

    print(f"Official Train size: {len(official_train_df)}")
    print(f"Official Dev size: {len(official_dev_df)}")

    # Augment training set
    final_train_df = aug.prepare_balanced_augmented_data(
        official_train_df,
        neg_to_pos_ratio=best_ratio,
        alpha=best_alpha,
        seed=42
    )
    print(f"Augmented Train size: {len(final_train_df)}")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    final_train_ds = du.TextDataset(final_train_df["text"], final_train_df["binary_label"], tokenizer)
    official_dev_ds = du.TextDataset(official_dev_df["text"], official_dev_df["binary_label"], tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].numpy()

        binary_preds = (probs >= best_threshold).astype(int)

        f1 = f1_score(labels, binary_preds, zero_division=0)
        prec = precision_score(labels, binary_preds, zero_division=0)
        rec = recall_score(labels, binary_preds, zero_division=0)
        acc = (binary_preds == labels).mean()

        tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()

        return {
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "accuracy": float(acc),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        }

    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2)
    class_weights = torch.tensor([1.0, best_pos_weight], dtype=torch.float)
    print("Using weights:", class_weights)

    training_args = TrainingArguments(
        output_dir=str(final_output_dir),
        eval_strategy="epoch",
        learning_rate=best_lr,
        per_device_train_batch_size=16,
        num_train_epochs=4,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none"
    )

    trainer = tr.WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=final_train_ds,
        eval_dataset=official_dev_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    trainer.train()

    print("Evaluation on dev set: ")
    dev_metrics = trainer.evaluate()

    print(f"F1        : {dev_metrics['eval_f1']:.4f}")
    print(f"Precision : {dev_metrics['eval_precision']:.4f}")
    print(f"Recall    : {dev_metrics['eval_recall']:.4f}")
    print("Confusion matrix:")
    print(f"TP  : {dev_metrics['eval_tp']}")
    print(f"TN  : {dev_metrics['eval_tn']}")
    print(f"FP  : {dev_metrics['eval_fp']}")
    print(f"FN  : {dev_metrics['eval_fn']}")
    print("="*50)

    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    print(f"Model saved to {final_output_dir}")

if __name__ == "__main__":
    main()