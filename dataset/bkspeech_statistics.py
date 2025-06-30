import os
import wave

folder_path = 'C:/Users/Barbara/Desktop/fer/DIPLOMSKI/dataset/BKSpeech/wavs' 

durations = []

for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        with wave.open(os.path.join(folder_path, filename), 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            durations.append(duration)

total_files = len(durations)
total_duration_sec = sum(durations)
total_duration_hr = total_duration_sec / 3600
avg_duration_sec = total_duration_sec / total_files
min_duration_sec = min(durations)
max_duration_sec = max(durations)

# Ispis rezultata
print(f'Broj datoteka: {total_files}')
print(f'Ukupno trajanje: {total_duration_hr:.2f} sati')
print(f'Prosječno trajanje: {avg_duration_sec:.2f} sekundi')
print(f'Minimalno trajanje: {min_duration_sec:.2f} sekundi')
print(f'Maksimalno trajanje: {max_duration_sec:.2f} sekundi')
