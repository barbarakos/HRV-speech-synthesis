import os
import librosa
import matplotlib.pyplot as plt

wav_dir = 'C:/Users/Barbara/Desktop/fer/DIPLOMSKI/dataset/BKSpeech/wavs' 

trajanja = []

for filename in os.listdir(wav_dir):
    if filename.endswith(".wav"):
        filepath = os.path.join(wav_dir, filename)
        try:
            y, sr = librosa.load(filepath, sr=None)
            trajanje = librosa.get_duration(y=y, sr=sr)
            trajanja.append(trajanje)
        except Exception as e:
            print(f"Greška kod {filename}: {e}")

plt.figure(figsize=(10, 6))
plt.hist(trajanja, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribucija trajanja zvučnih zapisa u BKSpeech korpusu')
plt.xlabel('Trajanje (sekunde)')
plt.ylabel('Broj zapisa')
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.tight_layout()
plt.show()
