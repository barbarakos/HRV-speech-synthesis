import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pystoi.stoi import stoi
from pesq import pesq

def get_wav_pairs(gt_dir, pred_dir):
    return [
        (os.path.join(gt_dir, f), os.path.join(pred_dir, f), f)
        for f in os.listdir(gt_dir)
        if f.endswith(".wav") and os.path.exists(os.path.join(pred_dir, f))
    ]

def load_and_prepare(path, sr=16000):
    audio, orig_sr = librosa.load(path, sr=None, mono=True)
    audio, _ = librosa.effects.trim(audio)  # trim silence
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio, sr

def main():
    gt_dir = "/mnt/d/fer/DIPLOMSKI/datasets/BKSpeech/test/wavs-test"
    pred_dir = "/mnt/d/fer/DIPLOMSKI/predictions/glowtts_hifigan/G133393_H220864"

    pairs = get_wav_pairs(gt_dir, pred_dir)
    results = []
    for gt_path, pred_path, fname in pairs:
        try:
            gt_audio, sr = load_and_prepare(gt_path)
            pred_audio, _ = load_and_prepare(pred_path)

            min_len = min(len(gt_audio), len(pred_audio))
            gt_audio = gt_audio[:min_len]
            pred_audio = pred_audio[:min_len]
            pesq_score = pesq(sr, gt_audio, pred_audio, 'wb')

            stoi_score = stoi(gt_audio, pred_audio, sr, extended=False)

        except Exception as e:
            print(f"Greška kod {fname}: {e}")
            pesq_score = np.nan
            stoi_score = np.nan

        print(f"{fname} | STOI: {stoi_score:.3f} | PESQ: {pesq_score:.2f}")
        results.append({
            "filename": fname,
            "STOI": stoi_score,
            "PESQ": pesq_score
        })

    df = pd.DataFrame(results)
    df.to_csv("tts_fixed_pesq_stoi_results.csv", index=False)
    print("\nRezultati spremljeni u 'tts_fixed_pesq_stoi_results.csv'")

if __name__ == "__main__":
    main()


