from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report

# Options: 'MR' or 'Semeval2017A'
# DATASET = 'Semeval2017A'
DATASET = 'MR'

MODEL_CONFIGS = {
    # For MR Dataset
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

    # For Semeval2017A Dataset
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

if __name__ == '__main__':
    # load the dataset
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_test = le.transform(y_test)

    models_to_run = {
        name: cfg for name, cfg in MODEL_CONFIGS.items() if cfg['dataset'] == DATASET
    }
    # run the experiments
    for model_name, cfg in models_to_run.items():
        print(f"\nRunning model: {model_name}")
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

        y_pred_labels = []
        for x in tqdm(X_test, desc=f"Inferencing with {model_name}"):
            result = sentiment_pipeline(x)[0]
            label = result['label']
            mapped_label = cfg['labels'][label]
            y_pred_labels.append(mapped_label)
        y_pred = le.transform(y_pred_labels)
        print(f'\nDataset: {DATASET}\nModel: {model_name}\nTest Set Evaluation:\n{get_metrics_report(y_test, y_pred)}')
