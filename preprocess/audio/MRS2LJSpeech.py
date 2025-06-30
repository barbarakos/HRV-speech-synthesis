from genericpath import exists
import glob
import sqlite3
import os
import argparse
import sys

from shutil import copyfile
from shutil import rmtree

cwd = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(cwd, "dataset")
output_dir_audio = ""
output_dir_audio_temp=""
output_dir_speech = ""

def create_folders():
  global output_dir
  global output_dir_audio
  global output_dir_audio_temp
  global output_dir_speech

  print('→ Creating Dataset Folders')

  output_dir_speech = os.path.join(output_dir, "BKSpeech")

  if os.path.exists(output_dir_speech):
    rmtree(output_dir_speech)

  output_dir_audio = os.path.join(output_dir_speech, "wavs")
  output_dir_audio_temp = os.path.join(output_dir_speech, "temp")

  os.makedirs(output_dir_speech)
  os.makedirs(output_dir_audio)
  os.makedirs(output_dir_audio_temp)

def convert_audio():
  global output_dir_audio
  global output_dir_audio_temp

  recordings = len([name for name in os.listdir(output_dir_audio_temp) if os.path.isfile(os.path.join(output_dir_audio_temp,name))])
  
  print('→ Converting %s Audio Files to 22050 Hz, 16 Bit, Mono and Normalizing to -23 LUFS\n' % "{:,}".format(recordings))

  import ffmpeg

  for idx, wav in enumerate(glob.glob(os.path.join(output_dir_audio_temp, "*.wav"))):

    percent = (idx + 1) / recordings

    print('› \033[96m%s\033[0m \033[2m%s / %s (%s)\033[0m ' % (os.path.basename(wav), "{:,}".format((idx + 1)), "{:,}".format(recordings), "{:.0%}".format(percent)))

    (ffmpeg
      .input(wav)
      .output(
          os.path.join(output_dir_audio, os.path.basename(wav)),
          acodec='pcm_s16le',
          ac=1,
          ar=22050,
          af='loudnorm=I=-23:TP=-2:LRA=11',
          loglevel='error')
      .overwrite_output()
      .run(capture_stdout=True)
    )

def copy_audio():
  global output_dir_audio

  print('→ Using ffmpeg to convert recordings')
  recordings = len([name for name in os.listdir(output_dir_audio_temp) if os.path.isfile(os.path.join(output_dir_audio_temp,name))])
  
  print('→ Copy %s Audio Files to LJSpeech Dataset\n' % "{:,}".format(recordings))

  for idx, wav in enumerate(glob.glob(os.path.join(output_dir_audio_temp, "*.wav"))):    
    copyfile(wav,os.path.join(output_dir_audio, os.path.basename(wav)))

import re

def clean_text(text):
    substitutions = {
        r'\btj\.?\b': 'to jest',
        r'\bp\.?s\.?\b': 'pees',
        r'\bwc\b': 'vece',
        r'\bpizze\b': 'pice',
        r'\bcartier\b': 'kartier',
        r'\bnpr\.?\b': 'na primjer'
    }

    text = text.lower()
    text = re.sub(r'(?<=\w)-(?=\w)', ' ', text)
    text = re.sub(r'[“”„"\'‘’‚‚()\[\]{}<>:;–—]', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()

    for pattern, replacement in substitutions.items():
        text = re.sub(pattern, replacement, text)

    return text

def create_meta_data(mrs_dir):
  print('→ Creating META Data')

  conn = sqlite3.connect(os.path.join(mrs_dir, "backend", "db", "mimicstudio.db"))
  c = conn.cursor()

  metadata = open(os.path.join(output_dir_speech, "metadata.csv"), mode="w", encoding="utf8")

  user_models = c.execute('SELECT uuid, user_name from usermodel ORDER BY created_date DESC').fetchall()
  user_id = user_models[0][0]

  for row in user_models:
    print(row[0] + ' -> ' + row[1])

  user_answer = input('Please choose ID of recording session to export (default is newest session) [' + user_id + ']: ')

  if user_answer:
    user_id = user_answer


  for row in c.execute('SELECT audio_id, prompt, lower(prompt) FROM audiomodel WHERE user_id = "' + user_id + '" ORDER BY length(prompt)'):
    source_file = os.path.join(mrs_dir, "backend", "audio_files", user_id, row[0] + ".wav")
    if exists(source_file):
      cleaned_text = clean_text(row[1])
      metadata.write(row[0] + "|" + row[1] + "|" + cleaned_text + "\n")
      copyfile(source_file, os.path.join(output_dir_audio_temp, row[0] + ".wav"))
    else:
      print("Wave file {} not found.".format(source_file))

  metadata.close()
  conn.close()

def cleanup():
  global output_dir_audio_temp

  rmtree(output_dir_audio_temp)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mrs_dir', required=True)
  parser.add_argument('--ffmpeg', required=False, default=False)
  args = parser.parse_args()
  
  if not os.path.isdir(os.path.join(args.mrs_dir,"backend")):
    sys.exit("Passed directory is no valid Mimic-Recording-Studio main directory!")

  print('\n\033[48;5;22m  MRS to LJ Speech Processor  \033[0m\n')

  create_folders()
  create_meta_data(args.mrs_dir)

  if(args.ffmpeg):
    convert_audio()
  
  else:
    copy_audio()
  
  cleanup()

  print('\n\033[38;5;86;1m✔\033[0m COMPLETE【ツ】\n')

if __name__ == '__main__':
  main()
