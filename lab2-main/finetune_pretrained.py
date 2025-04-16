import numpy as np
import evaluate
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.load_datasets import load_MR, load_Semeval2017A
from sklearn.metrics import classification_report
import argparse
import torch
from transformers import EarlyStoppingCallback


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
parser = argparse.ArgumentParser(description="Sentiment Analysis Transfer Learning")
parser.add_argument('--dataset', type=str, choices=['MR', 'Semeval2017A'], default='MR',
                    help="Dataset to use: 'MR' or 'Semeval2017A' (default: MR)")
args = parser.parse_args()
DATASET = args.dataset

metric = evaluate.load("accuracy")

MODEL_CONFIGS = {
    'siebert/sentiment-roberta-large-english': {
        'labels': {'POSITIVE': 'positive', 'NEGATIVE': 'negative'},
        'dataset': 'MR',
    },
    'textattack/bert-base-uncased-imdb': {
        'labels': {'LABEL_1': 'positive', 'LABEL_0': 'negative'},
        'dataset': 'MR',
    },
    'distilbert-base-uncased-finetuned-sst-2-english': {
        'labels': {'POSITIVE': 'positive', 'NEGATIVE': 'negative'},
        'dataset': 'MR',
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'labels': {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'},
        'dataset': 'Semeval2017A',
    },
    'cardiffnlp/twitter-xlm-roberta-base-sentiment': {
        'labels': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'dataset': 'Semeval2017A',
    },
    'finiteautomata/bertweet-base-sentiment-analysis': {
        'labels': {'NEG': 'negative', 'NEU': 'neutral', 'POS': 'positive'},
        'dataset': 'Semeval2017A',
    },
}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



def tokenize_function(examples):
    model_max_length = tokenizer.model_max_length if tokenizer.model_max_length is not None else 128
    model_max_length = min(tokenizer.model_max_length or 128, 512)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=model_max_length)

def prepare_dataset(X, y):
    return Dataset.from_dict({'text': X, 'label': y})


if __name__ == '__main__':
    # load raw dataset
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    n_classes = len(le.classes_)

    # split train/val sets
    X_train_split, X_valid, y_train_split, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    train_set = prepare_dataset(X_train_split, y_train_split)
    valid_set = prepare_dataset(X_valid, y_valid)
    test_set = prepare_dataset(X_test, y_test)
    results_summary = {}

    # execute the experiment
    for model_name, config in MODEL_CONFIGS.items():
        if config['dataset'] != DATASET:
            continue

        print(f"\nRunning model: {model_name} on {DATASET}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes).to(device)

        tokenized_train_set = train_set.map(tokenize_function)
        tokenized_valid_set = valid_set.map(tokenize_function)
        tokenized_test_set = test_set.map(tokenize_function)

        n_samples = 40
        small_train_dataset = tokenized_train_set.shuffle(
            seed=42).select(range(n_samples))
        small_eval_dataset = tokenized_test_set.shuffle(
            seed=42).select(range(n_samples))
        
        args = TrainingArguments(
            output_dir=f"output/{model_name.replace('/', '_')}",
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=1,
            learning_rate=1e-5,
            weight_decay=0.01,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

        trainer.train()
        predictions = trainer.predict(tokenized_test_set)
        preds = np.argmax(predictions.predictions, axis=-1)
        y_true = predictions.label_ids
        print(classification_report(y_true, preds))
