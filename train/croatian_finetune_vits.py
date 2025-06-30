import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from typing import List
from dataclasses import field
from TTS.tts.models.vits import Vits, VitsAudioConfig
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

audio_config = VitsAudioConfig(
   mel_fmin=95,
   mel_fmax=8000,
)

test_sentences = [
    ["Ovo je primjer sintetiziranja govora."],
    ["Dobar dan, kako ste danas?"],
    ["Zagreb je glavni grad Hrvatske."],
]

config = VitsConfig(
   audio=audio_config,
   batch_size=4,
   eval_batch_size=2,
   num_loader_workers=4,
   num_eval_loader_workers=1,
   test_delay_epochs=10,
   epochs=1000,
   text_cleaner="multilingual_cleaners",
   use_phonemes=True,
   phoneme_language="hr",
   phonemizer="espeak",
   phoneme_cache_path=os.path.join(output_path, "phoneme_cache_hr"),
   output_path=output_path,
   datasets=[dataset_config],
   test_sentences=test_sentences,
   eval_split_size=0.05,
   lr_gen=5e-5,
   lr_disc=5e-5,
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_size=config.eval_split_size,
)

model = Vits(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()