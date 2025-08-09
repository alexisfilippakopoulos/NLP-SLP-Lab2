import os
import warnings

from matplotlib import pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from attention import MultiHeadAttentionModel, SimpleSelfAttentionModel, TransformerEncoderModel
from config import EMB_PATH
from dataloading import SentenceDataset
from early_stopper import EarlyStopper
from models import LSTM, BaselineDNN
from training import get_metrics_report, torch_train_val_split, train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)  # EX1
y_test = label_encoder.transform(y_test)  # EX1
n_classes = label_encoder.classes_.size  # EX1 - LabelEncoder.classes_.size

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

train_loader, val_loader = torch_train_val_split(train_set, BATCH_SIZE, BATCH_SIZE)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = LSTM(n_classes, embeddings=embeddings)
# model = LSTM(n_classes, embeddings=embeddings, bidirectional=True)
# model = SimpleSelfAttentionModel(n_classes, embeddings=embeddings)
# model = MultiHeadAttentionModel(n_classes, embeddings=embeddings)
# model = TransformerEncoderModel(n_classes, embeddings=embeddings)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
criterion = torch.nn.CrossEntropyLoss()  # EX8
parameters = [p for p in model.parameters() if p.requires_grad]  # EX8
optimizer = torch.optim.Adam(parameters)  # EX8

train_losses = []
test_losses = []
#############################################################################
# Training Pipeline
#############################################################################
save_path = f'{DATASET}_{model.__class__.__name__}.pth'
early_stopper = EarlyStopper(model, save_path, patience=5) 
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)
    
    # evaluate the performance of the model, on both data sets
    valid_loss, (y_valid_gold, y_valid_pred) = eval_dataset(val_loader,
                                                            model,
                                                            criterion)
    
    print(f"\n===== EPOCH {epoch} ========")
    print(f'\nTraining set\n{get_metrics_report(y_train_gold, y_train_pred)}')
    print(f'\nValidation set\n{get_metrics_report(y_valid_gold, y_valid_pred)}')

    if early_stopper.early_stop(valid_loss):
        print('Early Stopping was activated.')
        print(f'Epoch {epoch}/{EPOCHS}, Loss at training set: {train_loss}\n\tLoss at validation set: {valid_loss}')
        print('Training has been completed.\n')
        break