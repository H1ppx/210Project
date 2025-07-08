import os
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class RadioDataset(Dataset):
    def __init__(self, file_paths, labels, n_fft=128, hop_length=64):
        self.file_paths = file_paths
        self.labels = labels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        x = np.load(self.file_paths[idx])
        x_iq = torch.tensor(np.stack([x.real, x.imag]), dtype=torch.float32)
        x_stft = torch.stft(x_iq, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        x_mag = x_stft.abs()
        x_combined = torch.mean(x_mag, dim=0, keepdim=True)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_combined, y

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 32 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            predicted = preds.argmax(1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

def load_data_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sample_path, modulation FROM signals WHERE modulation IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    paths = [r[0] for r in rows if os.path.isfile(r[0])]
    mods = [r[1] for r in rows if os.path.isfile(r[0])]
    return paths, mods

sample_paths, mod_labels = load_data_from_db("radioml_signals.db")
encoder = LabelEncoder()
label_ids = encoder.fit_transform(mod_labels)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    sample_paths, label_ids, test_size=0.2, stratify=label_ids, random_state=42
)

train_dataset = RadioDataset(train_paths, train_labels)
test_dataset = RadioDataset(test_paths, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(num_classes=len(encoder.classes_)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    loss = train(model, train_loader, optimizer, criterion, device)
    acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
