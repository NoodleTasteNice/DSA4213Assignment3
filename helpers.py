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
from transformers.trainer_utils import get_last_checkpoint

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

def eval_per_class(model, tokenizer, texts, labels, max_len=128):
    import numpy as np, torch
    device = next(model.parameters()).device
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for txt in texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True,
                            padding=True, max_length=max_len)
            enc = {k: v.to(device) for k,v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            preds.append(int(p.argmax()))
            probs.append(float(p[1]))  # prob of positive
    y_true = np.array(labels, dtype=int)
    y_pred = np.array(preds, dtype=int)
    print("\nPer-class report (0=neg, 1=pos):\n",
          classification_report(y_true, y_pred, digits=3))
    print("Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(y_true, y_pred))
    return y_pred, probs

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_args(run_name):
    return TrainingArguments(
        output_dir=f"./results_{run_name}",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        eval_strategy="epoch",
        report_to="none",
        logging_strategy='steps',
        save_strategy="epoch",          # ✅ saves model after each epoch
        load_best_model_at_end=True,    # ✅ restores best checkpoint automatically
        metric_for_best_model="eval_loss",  # or "eval_accuracy"
        greater_is_better=False, 
        fp16=torch.cuda.is_available(),
    )

def plot_learning_curves(trainer, title_prefix, out_png_prefix):
    """
    Plot eval loss vs epoch using ONLY the evals that happen during training.
    Excludes the post-training evaluate() and any best-model reload logging.
    """
    logs = trainer.state.log_history

    # Keep only eval logs that happened during training (have both 'epoch' and 'step')
    eval_logs = [e for e in logs if ("eval_loss" in e and "epoch" in e and "step" in e)]

    if not eval_logs:
        print("[plot] No in-training eval logs found (check evaluation_strategy='epoch').")
        return

    # Deduplicate epochs: keep the FIRST eval per epoch (training loop one)
    by_epoch = {}
    for e in eval_logs:
        ep = float(e["epoch"])
        if ep not in by_epoch:
            by_epoch[ep] = float(e["eval_loss"])

    # Sort by epoch
    epochs = sorted(by_epoch.keys())
    eval_losses = [by_epoch[ep] for ep in epochs]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(epochs, eval_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Eval loss")
    plt.title(f"{title_prefix}: Eval loss vs epochs")
    plt.grid(True, alpha=0.3)
    out2 = f"{out_png_prefix}_eval.png"
    plt.savefig(out2, bbox_inches="tight"); plt.close()
    print(f"[Saved] {out2}")

def collect_qualitative_errors(model, tokenizer, texts, labels, k=5, max_len=128):
    device = next(model.parameters()).device
    model.eval()
    mistakes = []
    with torch.no_grad():
        for txt, gold in zip(texts, labels):
            enc = tokenizer(txt, return_tensors="pt", truncation=True,
                            padding=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = F.softmax(model(**enc).logits, dim=-1).cpu().numpy()[0]
            pred = int(probs.argmax())
            if pred != int(gold):
                mistakes.append({
                    "text": txt,
                    "gold": int(gold),
                    "pred": pred,
                    "prob_pos": float(probs[1]),
                    "prob_neg": float(probs[0]),
                    "confidence": float(max(probs)),
                })
    # sort by confidence in wrong prediction (descending)
    mistakes = sorted(mistakes, key=lambda x: x["confidence"], reverse=True)
    return mistakes[:k]

def print_mistakes(mistakes, title):
    print(f"\n=== Qualitative Errors: {title} (showing {len(mistakes)}) ===")
    for j, m in enumerate(mistakes, 1):
        short = (m["text"][:300] + "…") if len(m["text"]) > 300 else m["text"]
        print(f"[{j}] gold={m['gold']} pred={m['pred']}  p_pos={m['prob_pos']:.2f}  p_neg={m['prob_neg']:.2f}")
        print(short)
        print("-" * 80)

def compare_full_vs_lora(full_model, lora_model, tokenizer, texts, labels, k=10, max_len=128):
    import numpy as np, pandas as pd, torch
    print("Starting model comparison on", len(texts), "examples...")
    def predict_all(model):
        preds, probs = [], []
        model.eval()
        with torch.no_grad():
            for txt in texts:
                enc = tokenizer(txt, return_tensors="pt", truncation=True,
                                padding=True, max_length=max_len)
                enc = {k: v.to(next(model.parameters()).device) for k,v in enc.items()}
                p = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()[0]
                preds.append(int(p.argmax())); probs.append(float(p[1]))
        return np.array(preds), np.array(probs)

    pf, cf = predict_all(full_model)
    pl, cl = predict_all(lora_model)

    df = pd.DataFrame({
        "text": texts,
        "gold": labels,
        "full_pred": pf,
        "full_p_pos": np.round(cf, 3),
        "lora_pred": pl,
        "lora_p_pos": np.round(cl, 3),
    })
    df["agree"] = (df["full_pred"] == df["lora_pred"])
    df["full_correct"] = (df["full_pred"] == df["gold"])
    df["lora_correct"] = (df["lora_pred"] == df["gold"])
    df["confidence_gap"] = (df["full_p_pos"] - df["lora_p_pos"]).abs()

    print("\nAgreement rate:", df["agree"].mean())
    print("Full better count:", ((df["full_correct"]==1) & (df["lora_correct"]==0)).sum())
    print("LoRA better count:", ((df["lora_correct"]==1) & (df["full_correct"]==0)).sum())

    # Top disagreements to inspect
    disagreements = df[df["full_correct"] != df["lora_correct"]]
    to_show = disagreements.sort_values("confidence_gap", ascending=False).head(k)
    print("\n=== Top disagreements (by confidence gap) ===")
    for i, row in to_show.iterrows():
        tshort = (row["text"][:300] + "…") if len(row["text"])>300 else row["text"]
        print(f"\nGold={row['gold']} | Full pred={row['full_pred']} (p_pos={row['full_p_pos']}) | "
              f"LoRA pred={row['lora_pred']} (p_pos={row['lora_p_pos']})")
        print(tshort)
    return df