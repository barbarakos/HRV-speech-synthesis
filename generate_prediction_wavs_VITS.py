import csv
import subprocess
import os

csv_path = "/mnt/d/fer/DIPLOMSKI/datasets/BKSpeech/test/metadata-test.csv"

vits_models = [
   ("1047203", "/mnt/d/fer/DIPLOMSKI/hrv-vits-kaggle-finetune/HRV-VITS-finetuning-to1070000/output/run-June-23-2025_07+59AM-dbf1a08a/best_model_1047203.pth", 
         "/mnt/d/fer/DIPLOMSKI/hrv-vits-kaggle-finetune/HRV-VITS-finetuning-to1070000/output/run-June-23-2025_07+59AM-dbf1a08a/config.json"),
   ("987203", "/mnt/d/fer/DIPLOMSKI/hrv-vits-kaggle-finetune/HRV-VITS-finetuning-to1010000/output/run-June-22-2025_06+01PM-dbf1a08a/best_model_987203.pth", 
         "/mnt/d/fer/DIPLOMSKI/hrv-vits-kaggle-finetune/HRV-VITS-finetuning-to1010000/output/run-June-22-2025_06+01PM-dbf1a08a/config.json"),
   ("908602", "/mnt/d/fer/DIPLOMSKI/hrv-vits-kaggle-finetune/HRV-VITS-finetuning-to950000/output/run-June-16-2025_09+44PM-dbf1a08a/best_model_908602.pth", 
         "/mnt/d/fer/DIPLOMSKI/hrv-vits-kaggle-finetune/HRV-VITS-finetuning-to950000/output/run-June-16-2025_09+44PM-dbf1a08a/config.json"),
]

output_base = "/mnt/d/fer/DIPLOMSKI/predictions/vits"

with open(csv_path, mode="r", encoding="utf-8") as csvfile:
   reader = list(csv.reader(csvfile, delimiter="|"))

total_jobs = len(vits_models) * len(reader)
job_num = 1

for row in reader:
   if len(row) < 2:
      continue
   file_id = row[0].strip()
   text = row[1].strip()

   if not text:
      print(f"⚠️ Prazan tekst za file ID {file_id}, preskačem.")
      continue

   for v_step, v_model, v_config in vits_models:
      output_dir = f"{output_base}/VITS{v_step}"
      os.makedirs(output_dir, exist_ok=True)
      out_path = os.path.join(output_dir, f"{file_id}.wav")

      cmd = [
         "tts",
         "--text", text,
         "--model_path", v_model,
         "--config_path", v_config,
         "--out_path", out_path,
      ]

      print(f"[{job_num}/{total_jobs}] 🔄 Generating: {out_path}")
      job_num += 1

      try:
         subprocess.run(cmd, check=True)
      except subprocess.CalledProcessError as e:
         print(f"❌ Greška kod VITS{v_step}, file {file_id}: {e}")