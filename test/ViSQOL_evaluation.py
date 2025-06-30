import os
import numpy as np
import pandas as pd
from visqol import Visqol 
from scipy.io import wavfile

def get_wav_pairs(gt_dir, pred_dir):
    return [
        (os.path.join(gt_dir, f), os.path.join(pred_dir, f), f)
        for f in sorted(os.listdir(gt_dir))
        if f.endswith('.wav') and os.path.exists(os.path.join(pred_dir, f))
    ]

def main():
    gt_dir = '/kaggle/input/groundtruth/wavs-test'
    pred_dir = '/kaggle/input/predictions/G133393_H231929'

    pairs = get_wav_pairs(gt_dir, pred_dir)
    print(f"Pronađeno {len(pairs)} parova za ViSQOL evaluaciju.\n")

    visqol_model = Visqol()  
    results = []

    for gt_path, pred_path, fname in pairs:
        try:
            _, ref = wavfile.read(gt_path)
            _, deg = wavfile.read(pred_path)

            score = visqol_model.compute(ref, deg, sr=16000) 

            print(f"{fname} → ViSQOL MOS-LQO: {score:.3f}")
            results.append({'filename': fname, 'visqol_mos_lqo': score})

        except Exception as e:
            print(f"⚠️ Greška pri {fname}: {e}")
            results.append({'filename': fname, 'visqol_mos_lqo': np.nan})

    df = pd.DataFrame(results)
    df.to_csv('visqol_evaluation.csv', index=False)

    print("\n--- REZULTATI ---")
    avg = np.nanmean(df['visqol_mos_lqo'])
    print(f"Prosječan ViSQOL MOS-LQO: {avg:.3f}")
    print("Spremio ispod: visqol_evaluation.csv")

if __name__ == "__main__":
    main()
