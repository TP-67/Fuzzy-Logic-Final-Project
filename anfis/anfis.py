import os
import sys
sys.path.append(os.path.abspath('../src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from dataset import *


# Create output folder
abs_dir = os.path.abspath('../output')
file_name = os.path.splitext(__file__)[0]
output_dir = os.path.join(abs_dir, file_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Build model
class AnimalClassifier(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim):
        super(AnimalClassifier, self).__init__()

        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.01)

        self.lin2 = nn.Linear(hid_dim, hid_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.01)

        self.lin3 = nn.Linear(hid_dim // 2, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu1(self.lin1(x))
        h = self.dropout1(h)
        h = self.relu2(self.lin2(h))
        h = self.dropout2(h)
        # output = self.lin3(h)
        output = self.sigmoid(self.lin3(h))

        return output


def get_num_correct(pred, label):
    return pred.argmax(dim=1).eq(label).sum().item()


def binary_acc(y_pred, label):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == label).sum().float()
    acc = correct_results_sum / label.shape[0]
    acc = torch.round(acc * 100)

    return acc


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
dataset = get_dataset('../data/animals.csv', output_dir)
train_set, test_set = torch.utils.data.random_split(dataset, [320, 80])

# Build dataloader
train_loader = get_dataloader(train_set, 8, 'train')
test_loader = get_dataloader(test_set, 8, 'test')

# Load models
model = AnimalClassifier(2, 32, 1).to(device)
print(model)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Criterion
criterion = nn.BCELoss()

# Training
epoch_list = []
train_loss_list = []
train_correct_list = []

for epoch in range(250):
    total_loss = 0
    total_correct = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        pred = model(features)
        loss = criterion(pred, labels.reshape(-1, 1))
        acc = binary_acc(pred, labels.reshape(-1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += acc.item()

    train_loss_list.append(total_loss / len(train_loader))
    train_correct_list.append(total_correct / len(train_loader))
    epoch_list.append(epoch)

    print(f'Epoch {epoch+0:03}: | Loss: {total_loss/len(train_loader):.5f} | Acc: {total_correct/len(train_loader):.3f}')


# Plots
plt.figure(1)
plt.plot(epoch_list, train_loss_list, label="Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.title('Training Loss')
plt.legend()
# Save image
plt.savefig(os.path.join(output_dir, 'loss.jpg'))
plt.show()

plt.figure(2)
plt.plot(epoch_list, train_correct_list, label="Acc")
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.title('Training Accuracy')
plt.legend()
# Save image
plt.savefig(os.path.join(output_dir, 'accuracy.jpg'))
plt.show()

# Testing
total_test_loss = 0
total_test_correct = 0
model.eval()
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        pred_test = model(features)
        loss = criterion(pred_test, labels.reshape(-1, 1))
        acc = binary_acc(pred_test, labels.reshape(-1, 1))

        total_test_loss += loss.item()
        total_test_correct += acc.item()

# Print testing results
output_steam = f'Testing Loss: {total_test_loss/len(test_loader):.5f} | Testing Acc: {total_test_correct/len(test_loader):.3f}'
print(output_steam)

# Output stream
with open(os.path.join(output_dir, 'testing_results.txt'), 'w') as text_file:
    text_file.write(output_steam)
    text_file.write('\n')
