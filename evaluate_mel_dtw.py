import os
import numpy as np
import pandas as pd
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def get_wav_pairs(gt_dir, pred_dir):
    return [
        (os.path.join(gt_dir, f), os.path.join(pred_dir, f), f)
        for f in os.listdir(gt_dir)
        if f.endswith(".wav") and os.path.exists(os.path.join(pred_dir, f))
    ]

def compute_mel_dtw(gt_audio, pred_audio, sr):
    # Normalizacija i izrezivanje
    if np.max(np.abs(gt_audio)) > 0:
        gt_audio = gt_audio / np.max(np.abs(gt_audio))
    if np.max(np.abs(pred_audio)) > 0:
        pred_audio = pred_audio / np.max(np.abs(pred_audio))

    min_len = min(len(gt_audio), len(pred_audio))
    gt_audio = gt_audio[:min_len]
    pred_audio = pred_audio[:min_len]

    # Mel-spektrogrami (T, 80)
    gt_mel = librosa.feature.melspectrogram(y=gt_audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    pred_mel = librosa.feature.melspectrogram(y=pred_audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)

    gt_mel_db = librosa.power_to_db(gt_mel).T
    pred_mel_db = librosa.power_to_db(pred_mel).T

    # DTW
    try:
        distance, _ = fastdtw(gt_mel_db, pred_mel_db, dist=euclidean)
        normalized_distance = distance / len(gt_mel_db)
    except Exception as e:
        print(f"⚠️ Greška u DTW: {e}")
        normalized_distance = np.nan

    return normalized_distance

def main():
    gt_dir = "/mnt/d/fer/DIPLOMSKI/datasets/BKSpeech/test/wavs-test"
    pred_dir = "/mnt/d/fer/DIPLOMSKI/predictions/glowtts_hifigan/G133393_H220864"

    pairs = get_wav_pairs(gt_dir, pred_dir)
    print(f"✅ Pronađeno {len(pairs)} parova .wav datoteka za evaluaciju.\n")

    results = []
    for gt_path, pred_path, fname in pairs:
        try:
            gt_audio, sr = librosa.load(gt_path, sr=None)
            pred_audio, _ = librosa.load(pred_path, sr=sr)

            mel_dtw = compute_mel_dtw(gt_audio, pred_audio, sr)
        except Exception as e:
            print(f"⚠️ Greška kod {fname}: {e}")
            mel_dtw = np.nan

        results.append({
            "filename": fname,
            "Mel_DTW": mel_dtw
        })
        print(f"{fname} | Mel-DTW: {mel_dtw:.2f}")

    df = pd.DataFrame(results)
    df.to_csv("mel_dtw_results.csv", index=False)

    print("\n--- 📊 PROSJEK ---")
    print(f"Avg. Mel-DTW: {np.nanmean(df['Mel_DTW']):.2f}")
    print("📁 Rezultati spremljeni u 'mel_dtw_results.csv'")

if __name__ == "__main__":
    main()
