# compare.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from helpers import compare_full_vs_lora   # or paste your helper here

# Load tokenizer (same as before)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load both models
full_model = AutoModelForSequenceClassification.from_pretrained("model_full")
lora_model = AutoModelForSequenceClassification.from_pretrained("model_lora")

# Load the same eval data
ds = load_dataset("imdb")
raw_eval = ds["test"].shuffle(seed=42).select(range(1000))
texts = raw_eval["text"]
labels = raw_eval["label"]

# Compare
compare_full_vs_lora(full_model, lora_model, tok, texts, labels, k=5)