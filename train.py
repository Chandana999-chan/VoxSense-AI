from pathlib import Path
import json
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from src.audio_features import (
    extract_features,
    extract_features_from_audio,
    load_audio,
)
from src.model import build_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
SAMPLE_RATE = 22050
MAX_EPOCHS = 40
BATCH_SIZE = 32

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ravdess"
ARTIFACTS_DIR = ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

# Fixed actor-independent split for comparison with baseline.
TRAIN_ACTORS = [
    2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 18, 21, 22, 23
]
VAL_ACTORS = [8, 13, 20, 24]
TEST_ACTORS = [1, 9, 12, 17, 19]


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------
# Dataset discovery
# --------------------------------------------------

def find_audio_files():
    """Find RAVDESS speech recordings and extract metadata."""

    records = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Check that the RAVDESS dataset is inside data/ravdess."
        )

    for path in sorted(DATA_DIR.rglob("*.wav")):
        parts = path.stem.split("-")

        if len(parts) != 7:
            continue

        try:
            modality = int(parts[0])
            vocal_channel = int(parts[1])
            emotion_code = int(parts[2])
            actor = int(parts[6])
        except ValueError:
            continue

        # RAVDESS: audio-only, speech, emotion codes 1–8.
        if modality != 3:
            continue

        if vocal_channel != 1:
            continue

        if emotion_code not in range(1, 9):
            continue

        records.append(
            {
                "path": str(path),
                "actor": actor,
                "emotion_code": emotion_code,
                "emotion": EMOTION_LABELS[emotion_code - 1],
            }
        )

    if not records:
        raise RuntimeError(
            f"No valid RAVDESS speech files found under {DATA_DIR}."
        )

    df = pd.DataFrame(records)

    if df["path"].duplicated().any():
        raise RuntimeError("Duplicate audio paths detected.")

    return df


# --------------------------------------------------
# Audio augmentation
# --------------------------------------------------

def augment_waveform(waveform, rng):
    """
    Apply mild training-only augmentation.

    Includes small additive noise and a zero-padded time shift.
    The waveform is not circularly wrapped.
    """

    audio = np.asarray(waveform, dtype=np.float32).copy()

    if audio.size == 0:
        return audio

    # Small additive noise relative to signal RMS.
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-8))
    noise_level = float(rng.uniform(0.0, 0.003))

    noise = rng.normal(
        loc=0.0,
        scale=rms * noise_level,
        size=audio.shape,
    ).astype(np.float32)

    audio += noise

    # Shift by at most approximately 40 ms.
    max_shift = int(0.04 * SAMPLE_RATE)
    shift = int(rng.integers(-max_shift, max_shift + 1))

    if shift != 0:
        shifted = np.zeros_like(audio)

        if shift > 0:
            shifted[shift:] = audio[:-shift]
        else:
            shifted[:shift] = audio[-shift:]

        audio = shifted

    return np.nan_to_num(audio).astype(np.float32)


# --------------------------------------------------
# Dataset preparation
# --------------------------------------------------

def prepare_dataset():
    """
    Load original waveforms and extract original features.

    Waveforms are retained so augmentation can be applied after
    the actor-independent split has been established.
    """

    df = find_audio_files()

    print(f"\nDiscovered recordings: {len(df)}")
    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().sort_index())

    waveforms = []
    features = []
    labels = []
    actors = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        waveform, feature, _ = extract_features_from_audio(row.path)

        waveforms.append(
            np.asarray(waveform, dtype=np.float32)
        )
        features.append(
            np.asarray(feature, dtype=np.float32)
        )
        labels.append(int(row.emotion_code) - 1)
        actors.append(int(row.actor))

        if i % 100 == 0:
            print(f"Processed {i}/{len(df)} recordings")

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(actors, dtype=np.int64)

    if X.ndim != 3:
        raise ValueError(f"Unexpected feature dimensions: {X.shape}")

    if X.shape[1] != 130 or X.shape[2] != 120:
        raise ValueError(
            f"Expected features shaped (N, 130, 120), got {X.shape}. "
            "Check src/audio_features.py."
        )

    if len(waveforms) != len(X):
        raise RuntimeError("Waveform and feature counts do not match.")

    df["emotion_index"] = y
    df.to_csv(
        ARTIFACTS_DIR / "dataset_manifest.csv",
        index=False,
    )

    return df, waveforms, X, y, groups


# --------------------------------------------------
# Fixed actor split
# --------------------------------------------------

def make_actor_splits(groups):
    """Create fixed train/validation/test indices by actor."""

    train_idx = np.where(np.isin(groups, TRAIN_ACTORS))[0]
    val_idx = np.where(np.isin(groups, VAL_ACTORS))[0]
    test_idx = np.where(np.isin(groups, TEST_ACTORS))[0]

    all_indices = np.concatenate([train_idx, val_idx, test_idx])

    if len(all_indices) != len(groups):
        missing = set(range(len(groups))) - set(all_indices.tolist())
        raise RuntimeError(
            f"Some recordings have actors outside the configured split: "
            f"{len(missing)} recordings."
        )

    if len(np.unique(all_indices)) != len(groups):
        raise RuntimeError("Dataset split contains duplicate indices.")

    train_group_set = set(groups[train_idx])
    val_group_set = set(groups[val_idx])
    test_group_set = set(groups[test_idx])

    assert train_group_set.isdisjoint(val_group_set)
    assert train_group_set.isdisjoint(test_group_set)
    assert val_group_set.isdisjoint(test_group_set)

    print("\nActor-independent split:")
    print("Train actors:", sorted(train_group_set))
    print("Validation actors:", sorted(val_group_set))
    print("Test actors:", sorted(test_group_set))

    print(
        f"Recordings — train: {len(train_idx)}, "
        f"validation: {len(val_idx)}, test: {len(test_idx)}"
    )

    split_info = {
        "train_actors": sorted(int(x) for x in train_group_set),
        "validation_actors": sorted(int(x) for x in val_group_set),
        "test_actors": sorted(int(x) for x in test_group_set),
        "train_recordings": int(len(train_idx)),
        "validation_recordings": int(len(val_idx)),
        "test_recordings": int(len(test_idx)),
        "split_type": "fixed actor-independent split",
        "random_seed": SEED,
    }

    with open(
        ARTIFACTS_DIR / "split_info.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(split_info, f, indent=2)

    return train_idx, val_idx, test_idx


# --------------------------------------------------
# Feature normalization
# --------------------------------------------------

def normalize_features(X, train_idx):
    """
    Fit normalization using original training recordings only.
    Reuse the resulting statistics for validation, test and
    augmented training recordings.
    """

    mean = X[train_idx].mean(
        axis=(0, 1),
        keepdims=True,
    )

    std = X[train_idx].std(
        axis=(0, 1),
        keepdims=True,
    )

    std = np.where(std < 1e-6, 1.0, std)

    X_normalized = (X - mean) / std
    X_normalized = np.nan_to_num(X_normalized).astype(np.float32)

    np.savez(
        ARTIFACTS_DIR / "feature_normalization.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )

    return X_normalized, mean, std


# --------------------------------------------------
# Training-only augmentation
# --------------------------------------------------

def build_augmented_training_set(
    waveforms,
    X_train_original,
    y_train,
    mean,
    std,
):
    """
    Keep each original training sample and add one augmented
    version. Validation and test data are never augmented.
    """

    rng = np.random.default_rng(SEED)

    augmented_features = []
    augmented_labels = []

    print("\nGenerating augmented training samples...")

    for i, waveform in enumerate(waveforms):
        augmented_waveform = augment_waveform(waveform, rng)

        augmented_feature = extract_features(augmented_waveform)
        augmented_feature = np.asarray(
            augmented_feature,
            dtype=np.float32,
        )

        if augmented_feature.shape != (130, 120):
            raise ValueError(
                "Augmented feature shape mismatch: "
                f"{augmented_feature.shape}"
            )

        augmented_features.append(augmented_feature)
        augmented_labels.append(y_train[i])

        if (i + 1) % 100 == 0:
            print(
                f"Augmented {i + 1}/{len(waveforms)} "
                "training recordings"
            )

    X_aug = np.stack(augmented_features).astype(np.float32)
    y_aug = np.asarray(augmented_labels, dtype=np.int64)

    X_aug = (X_aug - mean) / std
    X_aug = np.nan_to_num(X_aug).astype(np.float32)

    # Original samples + one augmented version per original sample.
    X_train = np.concatenate(
        [X_train_original, X_aug],
        axis=0,
    )

    y_train_combined = np.concatenate(
        [y_train, y_aug],
        axis=0,
    )

    # Shuffle only the training set.
    order = rng.permutation(len(X_train))
    X_train = X_train[order]
    y_train_combined = y_train_combined[order]

    print(f"Training samples after augmentation: {len(X_train)}")

    return X_train, y_train_combined


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def save_evaluation(y_true, probabilities, history):
    y_pred = np.argmax(probabilities, axis=1)

    test_accuracy = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(EMOTION_LABELS))),
        target_names=EMOTION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(EMOTION_LABELS))),
    )

    pd.DataFrame(
        cm,
        index=EMOTION_LABELS,
        columns=EMOTION_LABELS,
    ).to_csv(ARTIFACTS_DIR / "confusion_matrix.csv")

    with open(
        ARTIFACTS_DIR / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    metrics = {
        "experiment": "cnn_lstm_mfcc_delta_delta_augmented",
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "test_recordings": int(len(y_true)),
        "num_classes": int(len(EMOTION_LABELS)),
        "emotion_labels": EMOTION_LABELS,
        "epochs_completed": int(len(history.history["loss"])),
        "best_validation_loss": float(
            min(history.history["val_loss"])
        ),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "feature_type": "40 MFCC + 40 delta + 40 delta-delta",
        "feature_shape": [130, 120],
        "augmentation": "mild additive noise and zero-padded time shift",
        "split_type": "fixed actor-independent split",
    }

    with open(
        ARTIFACTS_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        ARTIFACTS_DIR / "training_history.csv",
        index=False,
    )

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(
        history.history["val_accuracy"],
        label="Validation",
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(
        history.history["val_loss"],
        label="Validation",
    )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        ARTIFACTS_DIR / "training_curves.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nFinal test results")
    print("------------------")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(EMOTION_LABELS))),
            target_names=EMOTION_LABELS,
            zero_division=0,
        )
    )

    print("\nSaved results in:", ARTIFACTS_DIR)


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def main():
    print("VoxSense AI — CNN-LSTM Training")
    print("=" * 40)

    df, waveforms, X, y, groups = prepare_dataset()

    train_idx, val_idx, test_idx = make_actor_splits(groups)

    # Keep a record of the original split before augmentation.
    X_normalized, mean, std = normalize_features(X, train_idx)

    X_train_original = X_normalized[train_idx]
    y_train_original = y[train_idx]

    X_val = X_normalized[val_idx]
    y_val = y[val_idx]

    X_test = X_normalized[test_idx]
    y_test = y[test_idx]

    # Build augmented training set from training actors only.
    train_waveforms = [waveforms[i] for i in train_idx]

    X_train, y_train = build_augmented_training_set(
        train_waveforms,
        X_train_original,
        y_train_original,
        mean,
        std,
    )

    print("\nFinal data shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    # The model must accept 120 features per frame.
    model = build_model(input_shape=(130, 120), num_classes=8)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ARTIFACTS_DIR / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Reload the best validation-loss checkpoint.
    best_model = tf.keras.models.load_model(
        ARTIFACTS_DIR / "best_model.keras"
    )

    # Evaluate the test set only after training is complete.
    probabilities = best_model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    save_evaluation(y_test, probabilities, history)


if __name__ == "__main__":
    main()