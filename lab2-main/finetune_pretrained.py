import numpy as np
import evaluate
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.load_datasets import load_MR, load_Semeval2017A

DATASET = 'MR'  # 'MR' or 'Semeval2017A'
PRETRAINED_MODEL = 'bert-base-cased'

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def prepare_dataset(X, y):
    return Dataset.from_dict({'text': X, 'label': y})


if __name__ == '__main__':
    # load the raw data
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

    # split train into train and validation
    X_train_split, X_valid, y_train_split, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # prepare datasets
    train_set = prepare_dataset(X_train_split, y_train_split)
    valid_set = prepare_dataset(X_valid, y_valid)
    test_set = prepare_dataset(X_test, y_test)

    # define model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL, num_labels=n_classes)

    # tokenize datasets
    tokenized_train_set = train_set.map(tokenize_function)
    tokenized_valid_set = valid_set.map(tokenize_function)
    tokenized_test_set = test_set.map(tokenize_function)

    # training setup
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,
        learning_rate=1e-7,                 
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train_set,
        eval_dataset=tokenized_valid_set,
        compute_metrics=compute_metrics,
    )

    # train
    trainer.train()

    # evaluate on test set
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test_set)
    print("Test Set Evaluation:", test_metrics)
