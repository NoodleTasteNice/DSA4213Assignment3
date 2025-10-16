import argparse, time, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from helpers import (compute_metrics, eval_per_class, count_trainable, make_args,
                      plot_learning_curves, collect_qualitative_errors, print_mistakes, 
                      compare_full_vs_lora)

def main(method):
    model_name = "distilbert-base-uncased"
    print(f"\n=== Running {method.upper()} fine-tuning ===\n")

    # Load IMDb and tokenize
    ds = load_dataset("imdb")
    tok = AutoTokenizer.from_pretrained(model_name)
    def tok_fn(b): return tok(b["text"], truncation=True, padding="max_length", max_length=128)
    tok_ds = ds.map(tok_fn, batched=True)
    tok_ds = tok_ds.rename_column("label", "labels")
    tok_ds.set_format("torch", columns=["input_ids","attention_mask","labels"])
    train_ds = tok_ds["train"].shuffle(seed=42).select(range(2000))
    eval_ds  = tok_ds["test"].shuffle(seed=42).select(range(1000))

    # also keep raw texts/labels (same subset order) for qualitative analysis
    raw_eval = ds["test"].shuffle(seed=42).select(range(1000))   # mirrors eval_ds
    raw_eval_texts  = raw_eval["text"]
    raw_eval_labels = raw_eval["label"]

    if method == "full":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    elif method == "lora":
        base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
            target_modules=["q_lin","k_lin","v_lin","out_lin"]
        )
        model = get_peft_model(base, cfg)
    else:
        raise ValueError("method must be 'full' or 'lora'")

    print("Trainable parameters:", f"{count_trainable(model):,}")
    args = make_args(method)
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=eval_ds,
                      tokenizer=tok, compute_metrics=compute_metrics)

    t0 = time.time(); trainer.train(); t1 = time.time()
    trainer.save_model(f"model_{method}")
    if trainer.args.load_best_model_at_end:
        trainer._load_best_model()
    res = trainer.evaluate()
    print(f"\n=== {method.upper()} RESULTS ===")
    print(f"Train time: {round(t1-t0,2)}s")
    print({k: round(v,4) for k,v in res.items()})

    history = trainer.state.log_history
    eval_logs = [x for x in history if "eval_accuracy" in x]

    if len(eval_logs) > 0:
        avg_acc = np.mean([x["eval_accuracy"] for x in eval_logs])
        avg_f1  = np.mean([x["eval_f1"] for x in eval_logs])
        avg_loss = np.mean([x["eval_loss"] for x in eval_logs])

        print("\n=== AVERAGE METRICS (across all epochs) ===")
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"Average Eval Loss: {avg_loss:.4f}")
    else:
        print("\nNo evaluation logs found — make sure evaluation_strategy='epoch'.")

    title = "Full FT (DistilBERT)" if method == "full" else "LoRA (DistilBERT)"
    out_prefix = f"curve_{method}"

    # Learning curves -> PNGs saved in your cwd
    plot_learning_curves(trainer, title, out_prefix)

    print("\n=== Per-class metrics (final/best model) ===")
    _ypred, _probs = eval_per_class(trainer.model, tok, raw_eval_texts, raw_eval_labels)

    # Qualitative errors -> prints 5 mismatches
    errs = collect_qualitative_errors(trainer.model, tok, raw_eval_texts, raw_eval_labels, k=5)
    print_mistakes(errs, title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="full",
                        help="'full' for full fine-tuning, 'lora' for parameter-efficient fine-tuning")
    args = parser.parse_args()
    main(args.method)