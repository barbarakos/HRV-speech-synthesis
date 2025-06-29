import os
import numpy as np
import pandas as pd
import speechmetrics

def get_wav_pairs(gt_dir, pred_dir):
    return [
        (os.path.join(gt_dir, f), os.path.join(pred_dir, f), f)
        for f in os.listdir(gt_dir)
        if f.endswith(".wav") and os.path.exists(os.path.join(pred_dir, f))
    ]

def main():
    gt_dir = "/kaggle/input/groundtruth/wavs-test"
    pred_dir = "/kaggle/input/predictions/G133393_H231929"

    pairs = get_wav_pairs(gt_dir, pred_dir)
    print(f"✅ Pronađeno {len(pairs)} parova za evaluaciju.\n")

    # Učitaj metric loader s MOSNet i PSQM/BSS eval... relativne
    metrics = speechmetrics.load(['mosnet','pesq','bsseval'], window=None)

    results = []
    for gt_path, pred_path, fname in pairs:
        try:
            print(f"\n🔍 Evaluacija {fname}...")

            # Ocijeni predikciju: relativno, pred, ref
            rel_scores = metrics(pred_path, gt_path)
            # Ocijeni GT samostalno: za absolute MOSNet
            abs_gt = metrics(gt_path)

            results.append({
                "filename": fname,
                "mosnet_ref_pred": rel_scores['mosnet'],
                "pesq": rel_scores.get('pesq', np.nan),
                "sdr": rel_scores.get('sisdr', np.nan),
                "mosnet_gt": abs_gt['mosnet']
            })
            print(f"📊 mos(pred↔gt): {rel_scores['mosnet']:.2f}, pesq: {rel_scores.get('pesq',np.nan):.2f}, sdr: {rel_scores.get('sisdr',np.nan):.2f}")
            print(f"🎯 mos(gt): {abs_gt['mosnet']:.2f}")

        except Exception as e:
            print(f"⚠️ Greška kod {fname}: {e}")
            results.append({
                "filename": fname,
                "mosnet_ref_pred": np.nan,
                "pesq": np.nan,
                "sdr": np.nan,
                "mosnet_gt": np.nan
            })

    df = pd.DataFrame(results)
    df.to_csv("mosnet_evaluation.csv", index=False)

    print("\n--- 📊 REZULTATI ---")
    print(f"Prosječan MOS(pred↔gt): {np.nanmean(df['mosnet_ref_pred']):.2f}")
    print(f"Prosječan PESQ: {np.nanmean(df['pesq']):.2f}")
    print(f"Prosječan SDR: {np.nanmean(df['sdr']):.2f}")
    print(f"Prosječan MOS(gt): {np.nanmean(df['mosnet_gt']):.2f}")
    print("📁 Rezultati spremljeni u 'mosnet_evaluation.csv'")

if __name__ == "__main__":
    main()
