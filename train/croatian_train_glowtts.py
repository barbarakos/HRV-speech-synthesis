import os
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseAudioConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


output_path = "/kaggle/working/output"

dataset_config = BaseDatasetConfig(
   formatter="ljspeech", 
   dataset_name="BKSpeech",
   path="/kaggle/input/bkspeech-train/BKSpeech-train/",
   meta_file_train="metadata-train.csv", 
   language="hr",
   phonemizer="espeak",
)

audio_config = BaseAudioConfig(
   mel_fmin=95.0,
   mel_fmax=8000.0,
   pitch_fmin=50.0,
   pitch_fmax=500.0,
)

test_sentences = [
    "Ovo je primjer sintetiziranja govora.",
    "Dobar dan, kako ste danas?",
    "Zagreb je glavni grad Hrvatske.",
]

config = GlowTTSConfig(
   audio=audio_config,
   batch_size=32,
   eval_batch_size=16,
   num_loader_workers=4,
   num_eval_loader_workers=1,
   test_delay_epochs=10,
   epochs=1000,
   save_step=5000,
   text_cleaner="multilingual_cleaners",
   use_phonemes=True,
   phoneme_language="hr",
   phonemizer="espeak",
   phoneme_cache_path=os.path.join(output_path, "phoneme_cache_hr"),
   output_path=output_path,
   datasets=[dataset_config],
   test_sentences=test_sentences,
   eval_split_size=0.05,
   grad_clip=1.0,
)

tokenizer, _ = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
   dataset_config,
   eval_split=True,
   eval_split_size=0.05,
)

ap = AudioProcessor.init_from_config(config)

model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
   TrainerArgs(), 
   config, 
   output_path, 
   model=model, 
   train_samples=train_samples, 
   eval_samples=eval_samples
)

trainer.fit()