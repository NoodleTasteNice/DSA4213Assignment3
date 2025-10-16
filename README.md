# DSA4213Assignment3
## About
This project explores the trade-offs between Full Fine-Tuning and Parameter-Efficient Fine-Tuning (LoRA) on the IMDb movie review dataset using DistilBERT as the base model.
The goal is to evaluate how LoRA compares to full fine-tuning in terms of training efficiency, performance, and generalisation, under limited hardware and resource constraints.

## Project Structure
### `compare.py`
script to print qualitative comparisons for both fine-tuned models.
### `helpers.py`
contains helper functions for `main.py` and `compare.py`
### `main.py`
main training and evaluation script
### `requirements.txt`
python dependencies 
### `README.md`
project documentation

## Project Setup
create and activate a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
install dependencies
```
pip install -r requirements.txt
```

## Dataset
The IMBb movie review dataset from Hugging Face contains 50,000 movie reviews. Due to hardware limitations, only 2000 reviews are used for model training and 1000 reviews are used for evaluation. 

## Model Setup
Two configurations were trained and compared:
- Full Fine-Tuning - all model parameters are updated.
- LoRA Fine-Tuning - only low-rank adapter matrices are trained, reducing the number of trainable parameters.

## Running the Experiment
For full fine-tuning: 
```
python main.py --method full
```
For LoRA:
```
python main.py --method lora
```
After training both models, qualitatively compare the two models by comparing reviews where they disagreed on the most.
```
python compare.py
```
