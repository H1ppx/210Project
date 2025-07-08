import os
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from model import CNNClassifier  # Assumes CNNClassifier is defined in model.py

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

def load_data_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sample_path, modulation FROM signals WHERE modulation IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    paths = [r[0] for r in rows if os.path.isfile(r[0])]
    mods = [r[1] for r in rows if os.path.isfile(r[0])]
    return paths, mods

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

def main():
    db_path = "radioml_signals.db"
    model_path = "cnn_model.pth"

    paths, labels = load_data_from_db(db_path)
    encoder = LabelEncoder()
    label_ids = encoder.fit_transform(labels)

    _, test_paths, _, test_labels = train_test_split(
        paths, label_ids, test_size=0.2, stratify=label_ids, random_state=42
    )

    test_dataset = RadioDataset(test_paths, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(num_classes=len(encoder.classes_
