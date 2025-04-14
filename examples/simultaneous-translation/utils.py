import numpy as np
from scipy.signal import resample
import logging

def decode_and_resample(audio_data, original_sample_rate, target_sample_rate):
    """Decodes audio data and resamples it to the target sample rate."""
    try:
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        num_original_samples = len(audio_np)
        num_target_samples = int(
            num_original_samples * target_sample_rate / original_sample_rate)
        resampled_audio = resample(audio_np, num_target_samples)
        return resampled_audio.astype(np.int16).tobytes()
    except Exception as e:
        logging.error(f"Error in resampling: {e}")
        return audio_data

def preprocess_realtime_text(text):
    """Preprocesses text received from the realtime transcription callback."""
    text = text.lstrip()
    if text.startswith("..."):
        text = text[3:]
    if text.endswith("...'."):
        text = text[:-1]
    if text.endswith("...'"):
        text = text[:-1]
    text = text.lstrip()
    if text:
        text = text[0].upper() + text[1:]
    return text 