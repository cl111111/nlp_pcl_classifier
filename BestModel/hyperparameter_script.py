import os, sys, gc
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import src.data_utils as du
import src.augment as aug
import src.eval as ev
import src.trainer as tr

def main():
    print("Initializing Grid Search...")

    user = 'cl6523'
    output_dir = f'/vol/bitbucket/{user}/nlp_project/BestModel/grid_outputs/'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "live_grid_results.csv")

    df = du.load_pcl_tsv_to_df(du.pcl_path)
    df = du.add_binary_label(df, threshold=2)
    train_ids, dev_ids = du.load_split_ids()
    train_df, dev_df = du.make_splits(df, train_ids, dev_ids)
    internal_train_df, internal_dev_df = du.create_internal_splits(train_df)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    internal_dev_ds = du.TextDataset(internal_dev_df["text"], internal_dev_df["binary_label"], tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    param_grid = {
        'alpha': [0.1, 0.15, 0.20, 0.25],
        'ratio': [2.5, 3.0, 3.5],
        'lr': [8e-6, 1e-5, 2e-5],
        'pos_weight': [1.5, 2.0, 2.5]
    }
    grid = list(ParameterGrid(param_grid))
    print(f"Total combinations to run: {len(grid)}")

    for i, params in enumerate(grid):
        run_name = f"run_{i+1:03d}"
        print(f"\n{'='*40}\nStarting {run_name}/{len(grid)}: {params}\n{'='*40}")

        curr_train_df = aug.prepare_balanced_augmented_data(
            internal_train_df,
            neg_to_pos_ratio=params['ratio'],
            alpha=params['alpha'],
            seed=42
        )
        curr_train_ds = du.TextDataset(curr_train_df["text"], curr_train_df["binary_label"], tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2)
        class_weights = torch.tensor([1.0, params['pos_weight']], dtype=torch.float)

        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, run_name),
            eval_strategy="epoch",
            learning_rate=params['lr'],
            per_device_train_batch_size=16,
            num_train_epochs=4,
            save_strategy="no",
            report_to="none"
        )

        trainer = tr.WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=curr_train_ds,
            eval_dataset=internal_dev_ds,
            data_collator=data_collator,
            compute_metrics=ev.compute_metrics,
            class_weights=class_weights
        )

        trainer.train()
        metrics = trainer.evaluate()

        run_result = params.copy()
        run_result['run_name'] = run_name
        run_result['best_f1'] = metrics.get("eval_f1", 0.0)
        run_result['best_threshold'] = metrics.get("eval_best_threshold", 0.5)

        temp_df = pd.DataFrame([run_result])
        if not os.path.isfile(csv_path):
            temp_df.to_csv(csv_path, index=False)
        else:
            temp_df.to_csv(csv_path, mode='a', header=False, index=False)

        print(f">> {run_name} F1: {run_result['best_f1']:.4f}")

        del model, trainer, curr_train_ds, curr_train_df
        torch.cuda.empty_cache()
        gc.collect()

    print("\nGrid search entirely finished! Check live_grid_results.csv!")

if __name__ == "__main__":
    main()