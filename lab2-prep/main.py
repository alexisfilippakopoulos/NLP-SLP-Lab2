import os
import warnings

from matplotlib import pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import get_metrics_report, train_dataset, eval_dataset
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
DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"

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
n_classes = label_encoder.classes_.size  # EX1 

print("First 10 labels and their corresponding integer values:")
print(list(zip(label_encoder.inverse_transform(y_train[:10]), y_train[:10])))

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

print("First 10 samles:")
print(train_set.data[:10])

print("First 5 samles in their original form and as returned by the SentenceDataset class:")
for i in range(5):
    original_dataitem = X_train[i]
    original_label = y_train[i]
    dataset_example, dataset_label, dataset_length = train_set[i]
    print(f"dataitem = \"{original_dataitem}\", label = \"{label_encoder.inverse_transform([original_label])[0]}\"")
    print(f"Return values: example = {dataset_example}, label = {dataset_label}, length = {dataset_length}")

# EX7 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = BaselineDNN(n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

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
# lists to acummulate train and test losses 
TRAIN_LOSS = []
TEST_LOSS = []
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    TRAIN_LOSS.append(train_loss)
    TEST_LOSS.append(test_loss)
    # compute metrics using sklearn functions
    print("Train loss:" , train_loss)
    print("Test loss:", test_loss)
    print("Train accuracy:" , accuracy_score(y_train_gold, y_train_pred))
    print("Test accuracy:" , accuracy_score(y_test_gold, y_test_pred))
    print("Train F1 score:", f1_score(y_train_gold, y_train_pred, average='macro'))
    print("Test F1 score:", f1_score(y_test_gold, y_test_pred, average='macro'))
    print("Train Recall:", recall_score(y_train_gold, y_train_pred, average='macro'))
    print("Test Recall:", recall_score(y_test_gold, y_test_pred, average='macro'))
# plot training and validation loss curves
    
plt.plot(range(1, EPOCHS + 1), TRAIN_LOSS, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), TEST_LOSS, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
