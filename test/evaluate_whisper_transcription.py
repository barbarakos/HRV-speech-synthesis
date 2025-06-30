import os
import librosa
import whisper
import pandas as pd
from jiwer import wer, cer

model = whisper.load_model("base")

def transcribe(path):
    result = model.transcribe(path, language="hr", fp16=False)
    return result["text"].strip()

def get_wav_pairs(gt_dir, pred_dir):
    pairs = []
    for fname in os.listdir(gt_dir):
        if fname.endswith(".wav") and os.path.exists(os.path.join(pred_dir, fname)):
            pairs.append((os.path.join(gt_dir, fname), os.path.join(pred_dir, fname), fname))
    return pairs

def main():
    gt_dir = "/putanja/do/groundtruth"
    pred_dir = "/putanja/do/predikcija"

    pairs = get_wav_pairs(gt_dir, pred_dir)
    print(f"Pronađeno {len(pairs)} parova za WER evaluaciju\n")

    results = []

    for gt_path, pred_path, fname in pairs:
        try:
            gt_text = transcribe(gt_path)
            pred_text = transcribe(pred_path)

            w = wer(gt_text, pred_text)
            c = cer(gt_text, pred_text)

            print(f"{fname} | WER: {w:.2%} | CER: {c:.2%}")
            results.append({"filename": fname, "GT": gt_text, "PRED": pred_text, "WER": w, "CER": c})

        except Exception as e:
            print(f"Greška u {fname}: {e}")
            results.append({"filename": fname, "GT": "", "PRED": "", "WER": None, "CER": None})

    df = pd.DataFrame(results)
    df.to_csv("whisper_wer_results.csv", index=False)

    avg_wer = df["WER"].mean()
    avg_cer = df["CER"].mean()
    print("\n--- REZULTATI ---")
    print(f"Prosječni WER: {avg_wer:.2%}")
    print(f"Prosječni CER: {avg_cer:.2%}")
    print("Rezultati spremljeni u 'whisper_wer_results.csv'")

if __name__ == "__main__":
    main()
