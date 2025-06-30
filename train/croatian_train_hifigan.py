from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.tts.configs.shared_configs import BaseAudioConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = "/kaggle/working/output"

audio_config = BaseAudioConfig(
   mel_fmin=95.0,
   mel_fmax=8000.0,
   pitch_fmin=50.0,
   pitch_fmax=500.0,
)

config = HifiganConfig(
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=1,
    test_delay_epochs=10,
    epochs=1000,
    eval_split_size=350,
    seq_len=4096,
    pad_short=2000,
    print_step=200,
    use_noise_augment=True,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path="/kaggle/input/hrv-dataset/BKSpeech/wavs/",
    output_path=output_path,
)


ap = AudioProcessor.init_from_config(config)

eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

model = GAN(config, ap)

trainer = Trainer(
    TrainerArgs(), 
    config, 
    output_path, 
    model=model, 
    train_samples=train_samples, 
    eval_samples=eval_samples
)

trainer.fit()