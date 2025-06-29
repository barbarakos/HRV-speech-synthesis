# f0 Pearson, f0 DTW, VUV Acc

import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def extract_f0_pyinf(audio, sr, fmin=75, fmax=600, frame_length=2048, hop_length=512):
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    vuv = voiced_flag.astype(int)
    return f0.flatten(), vuv.flatten()

def compute_prosody_metrics(gt_audio, pred_audio, sr):
    gt_f0, gt_vuv = extract_f0_pyinf(gt_audio, sr)
    pred_f0, pred_vuv = extract_f0_pyinf(pred_audio, sr)

    min_len = min(len(gt_f0), len(pred_f0))
    gt_f0, pred_f0 = gt_f0[:min_len], pred_f0[:min_len]
    gt_vuv, pred_vuv = gt_vuv[:min_len], pred_vuv[:min_len]

    # Pearson
    voiced_idx = np.logical_and(gt_f0 > 0, pred_f0 > 0)
    pearson = pearsonr(gt_f0[voiced_idx], pred_f0[voiced_idx])[0] if voiced_idx.sum() > 1 else np.nan

    # DTW
    try:
        gt_f0_list = gt_f0.reshape(-1).tolist()
        pred_f0_list = pred_f0.reshape(-1).tolist()
        dtw_distance, _ = fastdtw(gt_f0_list, pred_f0_list, dist=euclidean)
        dtw_norm = dtw_distance / len(gt_f0_list)
    except Exception as e:
        print(f"[WARN] DTW neuspješan: {e}")
        dtw_norm = np.nan

    # VUV
    vuv_acc = np.mean(gt_vuv == pred_vuv)

    return pearson, dtw_norm, vuv_acc

def get_wav_pairs(gt_dir, pred_dir):
    return [
        (os.path.join(gt_dir, f), os.path.join(pred_dir, f), f)
        for f in os.listdir(gt_dir)
        if f.endswith(".wav") and os.path.exists(os.path.join(pred_dir, f))
    ]

def main():
    gt_dir = "/kaggle/input/groundtruth/wavs-test"
    pred_dir = "/kaggle/input/predictions/G133393_H231929"
    target_sr = 22050

    pairs = get_wav_pairs(gt_dir, pred_dir)
    print(f"Pronađeno {len(pairs)} parova .wav datoteka za evaluaciju.\n")

    results = []
    for gt_path, pred_path, fname in pairs:
        try:
            gt_audio, _ = librosa.load(gt_path, sr=target_sr)
            pred_audio, _ = librosa.load(pred_path, sr=target_sr)

            pearson, dtw, vuv = compute_prosody_metrics(gt_audio, pred_audio, target_sr)
            print(f"{fname} | f0 Pearson: {pearson:.3f} | f0 DTW: {dtw:.2f} | VUV Acc: {vuv:.3f}")

            results.append({
                "filename": fname,
                "f0_pearson": pearson,
                "f0_dtw": dtw,
                "vuv_accuracy": vuv
            })
        except Exception as e:
            print(f"Greška u {fname}: {e}")
            results.append({
                "filename": fname,
                "f0_pearson": np.nan,
                "f0_dtw": np.nan,
                "vuv_accuracy": np.nan
            })

    df = pd.DataFrame(results)
    df.to_csv("f0_prosody_evaluation_pyinf.csv", index=False)

    print("\n--- REZULTATI ---")
    print(f"Prosječna f0 Pearson korelacija: {np.nanmean(df['f0_pearson']):.3f}")
    print(f"Prosječna f0 DTW udaljenost: {np.nanmean(df['f0_dtw']):.2f}")
    print(f"Prosječna VUV accuracy: {np.nanmean(df['vuv_accuracy']):.3f}")
    print("Rezultati spremljeni u 'f0_prosody_evaluation_pyinf.csv'")

if __name__ == "__main__":
    main()
