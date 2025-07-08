import os
import sqlite3
import datetime
import kagglehub
import numpy as np

DB_PATH = "radioml_signals.db"
DATA_FOLDER = "data/radioml_samples"
os.makedirs(DATA_FOLDER, exist_ok=True)

def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_path TEXT NOT NULL,
            modulation TEXT,
            snr INTEGER,
            center_freq REAL,
            sample_rate REAL,
            date_added TEXT
        );
    """)
    conn.commit()
    return conn, cursor

def insert_to_db(cursor, path, mod, snr, freq, rate):
    date = datetime.datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO signals (sample_path, modulation, snr, center_freq, sample_rate, date_added)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (path, mod, snr, freq, rate, date))

def extract_metadata_from_array_name(key):
    mod, snr_str = key.split("_")
    return mod, int(snr_str)

def main():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("nolasthitnotomorrow/radioml2016-deepsigcom")
    print("Path to dataset files:", path)

    data_file = os.path.join(path, "RML2016.10a_dict.npy")
    print("Loading dataset:", data_file)
    data_dict = np.load(data_file, allow_pickle=True).item()

    conn, cursor = connect_db(DB_PATH)

    center_freq = 2.4e9
    sample_rate = 1e6

    for key, samples in data_dict.items():
        mod, snr = extract_metadata_from_array_name(key)
        for i, sample in enumerate(samples):
            filename = f"{mod}_{snr}dB_{i:04d}.npy"
            full_path = os.path.join(DATA_FOLDER, filename)
            np.save(full_path, sample.astype(np.complex64))
            insert_to_db(cursor, full_path, mod, snr, center_freq, sample_rate)

    conn.commit()
    conn.close()
    print("All samples saved and metadata inserted.")

if __name__ == "__main__":
    main()
