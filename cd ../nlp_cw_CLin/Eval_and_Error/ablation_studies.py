import os
import sys
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import src.data_utils as du
import src.trainer as tr

def main():
    ablation_dir = PROJECT_ROOT / "Eval_and_Error" / "Ablation_Run"
    os.makedirs(ablation_dir, exist_ok=True)
    ablation_output_dir = ablation_dir / "model"
    os.makedirs(ablation_output_dir, exist_ok=True)

    csv_path = PROJECT_ROOT / "BestModel" / "grid_results.csv"
    grid_df = pd.read_csv(csv_path)
    best_run = grid_df.loc[grid_df['best_f1'].idxmax()]

    best_lr = float(best_run['lr'])
    best_pos_weight = float(best_run['pos_weight'])
    best_threshold = float(best_run['best_threshold'])

    print("Ablation")
    print(f"Using lr: {best_lr}")
    print(f"Using pos weight: {best_pos_weight}")
    print(f"Using threshold: {best_threshold}")
    print("Removing Data Augmentation")

    df = du.load_pcl_tsv_to_df(du.pcl_path)
    df = df.dropna(subset=['text', 'label'])
    df = du.add_binary_label(df, threshold=2)

    train_ids, dev_ids = du.load_split_ids()
    official_train_df, official_dev_df = du.make_splits(df, train_ids, dev_ids)

    print(f"Ablation train size: {len(official_train_df)}")
    print(f"Official dev size: {len(official_dev_df)}")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    ablation_train_ds = du.TextDataset(official_train_df["text"], official_train_df["binary_label"], tokenizer)
    official_dev_ds = du.TextDataset(official_dev_df["text"], official_dev_df["binary_label"], tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Applying the custom threshold
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        binary_preds = (probs >= best_threshold).astype(int)

        f1 = f1_score(labels, binary_preds, zero_division=0)
        prec = precision_score(labels, binary_preds, zero_division=0)
        rec = recall_score(labels, binary_preds, zero_division=0)
        acc = (binary_preds == labels).mean()

        cm = confusion_matrix(labels, binary_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        return {
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "acc": float(acc),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        }

    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2)
    class_weights = torch.tensor([1.0, best_pos_weight], dtype=torch.float)

    training_args = TrainingArguments(
        output_dir=str(ablation_dir),
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
        train_dataset=ablation_train_ds,
        eval_dataset=official_dev_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    trainer.train()

    print("Ablation results (No augmentation)\n")
    dev_metrics = trainer.evaluate()
    for key in ['eval_f1', 'eval_precision', 'eval_recall', 'eval_acc',
                'eval_tp', 'eval_tn', 'eval_fp', 'eval_fn']:
        print(f"{key.replace('eval_', '').upper():<10}: {dev_metrics[key]:.4f}")

    trainer.save_model(str(ablation_output_dir))
    print(f"Ablation model saved to {ablation_output_dir}")

if __name__ == "__main__":
    main()