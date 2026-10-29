
"""
Regularized CNN-LSTM model for VoxSense AI.
"""

import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    LSTM,
    Dense,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model


def build_model(
    input_shape=(130, 120),
    num_classes=8,
):
    inputs = Input(
        shape=input_shape,
        name="audio_features",
    )

    x = Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
    )(inputs)

    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.35)(x)

    x = Conv1D(
        96,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
    )(x)

    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.40)(x)

    x = LSTM(
        48,
        dropout=0.25,
        recurrent_dropout=0.0,
        kernel_regularizer=regularizers.l2(0.001),
    )(x)

    x = Dense(
        48,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
    )(x)

    x = Dropout(0.40)(x)

    # Softmax returns one probability for each emotion class.
    outputs = Dense(
        num_classes,
        activation="softmax",
        name="emotion_probabilities",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="VoxSense_CNN_LSTM_v2",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

