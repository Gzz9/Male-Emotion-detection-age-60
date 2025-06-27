import tkinter as tk
from tkinter import filedialog
import numpy as np
import librosa
import tensorflow as tf
import pickle

# === Load Models ===
gender_model = tf.keras.models.load_model("gender_model.keras")
age_model = tf.keras.models.load_model("age_binary_model.keras")
emotion_model = tf.keras.models.load_model("emotion_model.keras")

with open("label_encoder.pkl", "rb") as f:
    emotion_encoder = pickle.load(f)

# === Feature Extraction ===

def extract_mfcc_sequence(file_path, max_len=160):
    """For gender and age models: MFCC sequence of shape (160, 13)."""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc.T  # shape: (160, 13)
    except Exception as e:
        print(f"[ERROR] MFCC extraction failed: {e}")
        return None

def extract_mel_features(file_path, max_len=160):
    """For emotion model: Mel spectrogram of shape (128, 160, 1)."""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < max_len:
            pad_width = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :max_len]
        mel_db = mel_db[..., np.newaxis]  # shape: (128, 160, 1)
        return mel_db
    except Exception as e:
        print(f"[ERROR] Mel feature extraction failed: {e}")
        return None

# === Main Analysis Pipeline ===
def analyze_audio(file_path):
    try:
        # Step 1: Gender Detection
        mfcc_seq = extract_mfcc_sequence(file_path)
        if mfcc_seq is None:
            return "❌ Error: Could not extract MFCC sequence."
        X_mfcc = np.expand_dims(mfcc_seq, axis=0)  # shape: (1, 160, 13)

        gender_pred = gender_model.predict(X_mfcc)
        gender = "male" if np.argmax(gender_pred) == 1 else "female"
        print(f"[INFO] Gender: {gender}")
        if gender == "female":
            return "🚫 Rejected: Female voice detected."

        # Step 2: Age Prediction
        age_pred = age_model.predict(X_mfcc)
        is_senior = np.argmax(age_pred) == 1
        print(f"[INFO] Age Group: {'senior' if is_senior else 'non-senior'}")
        if not is_senior:
            return "🚫 Rejected: Not a senior (under 60)."

        # Step 3: Emotion Detection
        mel_feat = extract_mel_features(file_path)
        if mel_feat is None:
            return "❌ Error: Could not extract Mel features."
        X_mel = np.expand_dims(mel_feat, axis=0)  # shape: (1, 128, 160, 1)

        emotion_pred = emotion_model.predict(X_mel)
        emotion_label = emotion_encoder.inverse_transform([np.argmax(emotion_pred)])[0]
        print(f"[INFO] Emotion: {emotion_label}")
        return f"✅ Detected Emotion: {emotion_label}"

    except Exception as e:
        print(f"[ERROR] analyze_audio(): {e}")
        return f"❌ Error: {e}"

# === GUI Setup ===
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        result_label.config(text="🔄 Processing...", fg="orange")
        root.update_idletasks()
        result = analyze_audio(file_path)
        result_label.config(text=result, fg="green" if "✅" in result else "red")

root = tk.Tk()
root.title("🎙️ Senior Male Emotion Detector")
root.geometry("500x300")
root.resizable(False, False)

title = tk.Label(root, text="Upload a .wav file to detect emotion (for senior males)", font=("Helvetica", 14))
title.pack(pady=20)

browse_btn = tk.Button(root, text="Browse Audio File", command=browse_file, font=("Helvetica", 12), width=25)
browse_btn.pack()

result_label = tk.Label(root, text="", font=("Helvetica", 14), wraplength=400, justify="center")
result_label.pack(pady=30)

root.mainloop()
