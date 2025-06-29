import os
import numpy as np
import pandas as pd
import librosa
import torch
from scipy.spatial.distance import cosine
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Učitaj pretreniran wav2vec2 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

def get_embedding(audio, sr=16000):
    # Resamplaj i pripremi audio
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    audio = audio / np.max(np.abs(audio))  # normalizacija

    # Tokenizacija i ekstrakcija embeddinga
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state.squeeze(0)  # (T, D)

    # Srednji embedding (možeš koristiti i medijan, maxpool itd.)
    return hidden_states.mean(dim=0).numpy()

def compute_embedding_similarity(gt_audio, pred_audio, sr):
    emb_gt = get_embedding(gt_audio, sr)
    emb_pred = get_embedding(pred_audio, sr)
    sim = 1 - cosine(emb_gt, emb_pred)
    return sim

def get_wav_pairs(gt_dir, pred_dir):
    pairs = []
    for fname in os.listdir(gt_dir):
        if fname.endswith(".wav"):
            gt_path = os.path.join(gt_dir, fname)
            pred_path = os.path.join(pred_dir, fname)
            if os.path.exists(pred_path):
                pairs.append((gt_path, pred_path, fname))
    return pairs

def main():
    # 🔁 Postavi svoje putanje ovdje
    gt_dir = "/mnt/d/fer/DIPLOMSKI/datasets/BKSpeech/test/wavs-test"
    pred_dir = "/mnt/d/fer/DIPLOMSKI/predictions/glowtts_hifigan/G133393_H220864"

    pairs = get_wav_pairs(gt_dir, pred_dir)
    print(f"✅ Pronađeno {len(pairs)} parova .wav datoteka za evaluaciju.\n")

    results = []

    for gt_path, pred_path, fname in pairs:
        try:
            gt_audio, sr = librosa.load(gt_path, sr=None, mono=True)
            pred_audio, _ = librosa.load(pred_path, sr=sr, mono=True)

            sim = compute_embedding_similarity(gt_audio, pred_audio, sr)
            print(f"{fname} | Embedding cosine sim: {sim:.3f}")
        except Exception as e:
            print(f"⚠️ Greška u {fname}: {e}")
            sim = np.nan

        results.append({"filename": fname, "embedding_cosine_similarity": sim})

    df = pd.DataFrame(results)
    df.to_csv("embedding_similarity_results.csv", index=False)

    print("\n--- 📊 REZULTATI ---")
    print(f"Prosječna cosine sličnost: {np.nanmean(df['embedding_cosine_similarity']):.3f}")
    print("📁 Rezultati spremljeni u 'embedding_similarity_results.csv'")

if __name__ == "__main__":
    main()
