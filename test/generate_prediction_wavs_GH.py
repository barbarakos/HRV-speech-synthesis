import csv
import subprocess
import os

csv_path = "/mnt/d/fer/DIPLOMSKI/datasets/BKSpeech/test/metadata-test.csv"

glowtts_models = [
    ("101253", "/mnt/d/fer/DIPLOMSKI/hrv-glowtts-kaggle/HRV-GlowTTS-to130000/output/run-June-15-2025_03+57PM-dbf1a08a/best_model_101253.pth", 
               "/mnt/d/fer/DIPLOMSKI/hrv-glowtts-kaggle/HRV-GlowTTS-to130000/output/run-June-15-2025_03+57PM-dbf1a08a/config.json"),
    ("133393", "/mnt/d/fer/DIPLOMSKI/hrv-glowtts-kaggle/HRV-GlowTTS-to195000/output/run-June-16-2025_06+16AM-dbf1a08a/best_model_133393.pth", 
               "/mnt/d/fer/DIPLOMSKI/hrv-glowtts-kaggle/HRV-GlowTTS-to195000/output/run-June-16-2025_06+16AM-dbf1a08a/config.json"),
]

hifigan_models = [
    ("208521", "/mnt/d/fer/DIPLOMSKI/hrv-hifigan-kaggle/hrv_hifigan-to210000/output/run-June-21-2025_10+49PM-dbf1a08a/best_model_208521.pth",
               "/mnt/d/fer/DIPLOMSKI/hrv-hifigan-kaggle/hrv_hifigan-to210000/output/run-June-21-2025_10+49PM-dbf1a08a/config.json"),
    ("220864", "/mnt/d/fer/DIPLOMSKI/hrv-hifigan-kaggle/hrv_hifigan-to220000/output/run-June-22-2025_05+44PM-dbf1a08a/best_model_220864.pth",
               "/mnt/d/fer/DIPLOMSKI/hrv-hifigan-kaggle/hrv_hifigan-to220000/output/run-June-22-2025_05+44PM-dbf1a08a/config.json"),
    ("231929", "/mnt/d/fer/DIPLOMSKI/hrv-hifigan-kaggle/hrv_hifigan-to230000/output/run-June-23-2025_06+56AM-dbf1a08a/best_model_231929.pth",
               "/mnt/d/fer/DIPLOMSKI/hrv-hifigan-kaggle/hrv_hifigan-to230000/output/run-June-23-2025_06+56AM-dbf1a08a/config.json")
]

output_base = "/mnt/d/fer/DIPLOMSKI/predictions/glowtts_hifigan"

with open(csv_path, mode="r", encoding="utf-8") as csvfile:
    reader = list(csv.reader(csvfile, delimiter="|"))

total_jobs = len(glowtts_models) * len(hifigan_models) * len(reader)
job_num = 1

for row in reader:
    if len(row) < 2:
        continue
    file_id = row[0].strip()
    text = row[1].strip()

    if not text:
        print(f"Prazan tekst za file ID {file_id}, preskačem.")
        continue

    for g_step, g_model, g_config in glowtts_models:
        for h_step, h_model, h_config in hifigan_models:
            output_dir = f"{output_base}/G{g_step}_H{h_step}"
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{file_id}.wav")

            cmd = [
                "tts",
                "--text", text,
                "--model_path", g_model,
                "--config_path", g_config,
                "--vocoder_path", h_model,
                "--vocoder_config_path", h_config,
                "--out_path", out_path,
            ]

            print(f"[{job_num}/{total_jobs}] Generating: {out_path}")
            job_num += 1

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Greška kod G{g_step}_H{h_step}, file {file_id}: {e}")