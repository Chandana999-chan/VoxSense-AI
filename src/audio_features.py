"""
Audio preprocessing and feature extraction for VoxSense AI.

Features:
- 40 MFCC coefficients
- 40 delta MFCC coefficients
- 40 delta-delta MFCC coefficients

Output shape: (130, 120)
"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 130
MAX_DURATION_SECONDS = 3.0


def load_audio(
    source,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Load an audio path or file-like object as mono audio."""

    if isinstance(source, (str, bytes, Path)):
        waveform, sr = librosa.load(
            str(source),
            sr=sample_rate,
            mono=True,
        )
    else:
        if hasattr(source, "seek"):
            source.seek(0)

        audio_bytes = source.read()

        waveform, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=False,
        )

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        if sr != sample_rate:
            waveform = librosa.resample(
                waveform.astype(np.float32),
                orig_sr=sr,
                target_sr=sample_rate,
            )

    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = np.nan_to_num(waveform)

    if waveform.size == 0:
        raise ValueError("The supplied audio file is empty.")

    return waveform, sample_rate


def extract_features(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract MFCC, delta and delta-delta features.

    Returns an array with shape (130, 120).
    """

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    ).reshape(-1)

    if waveform.size == 0:
        raise ValueError("Cannot extract features from empty audio.")

    waveform = np.nan_to_num(waveform)

    target_length = int(MAX_DURATION_SECONDS * sample_rate)

    if len(waveform) > target_length:
        waveform = waveform[:target_length]

    elif len(waveform) < target_length:
        waveform = np.pad(
            waveform,
            (0, target_length - len(waveform)),
        )

    peak = np.max(np.abs(waveform))

    if peak > 1e-8:
        waveform = waveform / peak

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512,
    )

    delta = librosa.feature.delta(
        mfcc,
        width=9,
        order=1,
        mode="nearest",
    )

    delta_delta = librosa.feature.delta(
        mfcc,
        width=9,
        order=2,
        mode="nearest",
    )

    combined = np.concatenate(
        [mfcc, delta, delta_delta],
        axis=0,
    ).T

    if combined.shape[0] < MAX_FRAMES:
        combined = np.pad(
            combined,
            (
                (0, MAX_FRAMES - combined.shape[0]),
                (0, 0),
            ),
            mode="constant",
        )
    else:
        combined = combined[:MAX_FRAMES]

    return combined.astype(np.float32)


def extract_features_from_audio(
    source,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return waveform, extracted features and sample rate."""

    waveform, sample_rate = load_audio(source)

    features = extract_features(
        waveform,
        sample_rate,
    )

    return waveform, features, sample_rate

