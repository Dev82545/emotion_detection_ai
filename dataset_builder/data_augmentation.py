'''
This file contains fuctions to augment the data to prevent overfitting and model generalization. 
The augmentations include adding noise, pitch shifting and time stretching.
'''
import numpy as np
import librosa
import random


def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)


def time_stretch(y, rate=1.1):
    return librosa.effects.time_stretch(y, rate=rate)

def augment_audio(y, sr):
    """Apply ONE random augmentation"""
    choice = random.choice(["stretch", "pitch", "noise", "none"])

    if choice == "stretch":
        return time_stretch(y)
    elif choice == "pitch":
        return pitch_shift(y, sr)
    elif choice == "noise":
        return add_noise(y)
    else:
        return y