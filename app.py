"""
VoxSense AI
Speech Emotion Recognition using CNN-LSTM

Features:
- Dark-themed dashboard
- Model metrics and training history
- WAV audio upload and playback
- CNN-LSTM emotion prediction
- Emotion probability visualization
- Waveform visualization
- MFCC, delta, and delta-delta heatmaps
- CSV downloads
- Session-based prediction history

Project: VoxSense AI
Model: CNN-LSTM
Dataset: RAVDESS
"""

from pathlib import Path
import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from src.audio_features import extract_features_from_audio


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "best_model.keras"
NORMALIZATION_PATH = ARTIFACTS_DIR / "feature_normalization.npz"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
HISTORY_PATH = ARTIFACTS_DIR / "training_history.csv"
CONFUSION_PATH = ARTIFACTS_DIR / "confusion_matrix.csv"
REPORT_PATH = ARTIFACTS_DIR / "classification_report.json"
SPLIT_PATH = ARTIFACTS_DIR / "split_info.json"

EXPECTED_FRAMES = 130
EXPECTED_FEATURES = 120

EMOTIONS = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised",
]

EMOTION_COLORS = {
    "Neutral": "#94A3B8",
    "Calm": "#38BDF8",
    "Happy": "#FBBF24",
    "Sad": "#818CF8",
    "Angry": "#F87171",
    "Fearful": "#C084FC",
    "Disgust": "#4ADE80",
    "Surprised": "#FB923C",
}

PLOT_TEMPLATE = "plotly_dark"


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VoxSense AI | Speech Emotion Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CUSTOM DARK THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --background: #080B14;
        --surface: #111827;
        --surface-light: #172033;
        --border: rgba(148, 163, 184, 0.16);
        --text: #F1F5F9;
        --muted: #94A3B8;
        --accent: #8B5CF6;
        --accent-blue: #38BDF8;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 5% 0%,
                rgba(124, 58, 237, 0.14),
                transparent 30%
            ),
            radial-gradient(
                circle at 95% 10%,
                rgba(14, 165, 233, 0.10),
                transparent 25%
            ),
            #080B14;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 11, 20, 0.75);
    }

    [data-testid="stSidebar"] {
        background: #0B1020;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    h1, h2, h3 {
        color: #F8FAFC !important;
        letter-spacing: -0.035em;
    }

    p, label, span {
        color: inherit;
    }

    .hero-card {
        padding: 30px;
        border-radius: 24px;
        border: 1px solid rgba(139, 92, 246, 0.30);
        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.17),
                rgba(14, 165, 233, 0.07)
            ),
            rgba(17, 24, 39, 0.90);
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.20);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.12;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(
            90deg,
            #C4B5FD,
            #7DD3FC,
            #F0ABFC
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #CBD5E1;
        font-size: 1rem;
        line-height: 1.8;
        margin-top: 14px;
        max-width: 850px;
    }

    .eyebrow {
        color: #A78BFA;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .section-label {
        color: #A5B4FC;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        font-weight: 700;
        text-transform: uppercase;
    }

    .glass-card {
        background: rgba(17, 24, 39, 0.78);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 14px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            rgba(23, 32, 51, 0.96),
            rgba(13, 19, 34, 0.96)
        );
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px 18px;
        min-height: 135px;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-caption {
        color: #64748B;
        font-size: 0.75rem;
        margin-top: 10px;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(74, 222, 128, 0.10);
        border: 1px solid rgba(74, 222, 128, 0.30);
        color: #86EFAC;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .prediction-card {
        padding: 26px;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(139, 92, 246, 0.15),
                rgba(56, 189, 248, 0.08)
            ),
            #111827;
        border: 1px solid rgba(139, 92, 246, 0.28);
    }

    .prediction-emotion {
        font-size: 2.4rem;
        font-weight: 800;
        color: #E9D5FF;
        margin: 8px 0;
    }

    .muted {
        color: #94A3B8;
        font-size: 0.88rem;
        line-height: 1.7;
    }

    .small-note {
        color: #64748B;
        font-size: 0.75rem;
        line-height: 1.6;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(139, 92, 246, 0.55);
        border-radius: 16px;
        padding: 10px;
        background: rgba(17, 24, 39, 0.55);
    }

    div[data-testid="stButton"] button {
        border-radius: 12px;
        border: 1px solid rgba(139, 92, 246, 0.40);
        background: linear-gradient(
            90deg,
            #7C3AED,
            #6366F1
        );
        color: white;
        font-weight: 700;
        min-height: 44px;
    }

    div[data-testid="stDownloadButton"] button {
        border-radius: 10px;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.12);
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def show_hero(title, subtitle, eyebrow="VOXSENSE AI"):
    """Display the reusable dashboard hero section."""

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_metric(label, value, caption=""):
    """Display a reusable metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig, height=360):
    """Apply a consistent dark theme to Plotly figures."""

    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color="#CBD5E1",
            size=12,
        ),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zerolinecolor="rgba(148,163,184,0.10)",
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zerolinecolor="rgba(148,163,184,0.10)",
    )

    return fig


def read_json(path, default=None):
    """Read JSON safely."""

    if not path.exists():
        return {} if default is None else default

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def get_metric_value(metrics, keys, default=None):
    """
    Retrieve a metric using several possible key names.

    This accommodates small differences in metrics.json formats.
    """

    for key in keys:
        if key in metrics:
            value = metrics[key]

            if isinstance(value, dict):
                for nested_key in (
                    "test",
                    "test_accuracy",
                    "accuracy",
                    "value",
                ):
                    if nested_key in value:
                        return value[nested_key]

            return value

    return default


def format_percentage(value):
    """Format a metric as a percentage when possible."""

    if value is None:
        return "N/A"

    try:
        value = float(value)

        if not np.isfinite(value):
            return "N/A"

        if 0 <= value <= 1:
            value *= 100

        return f"{value:.2f}%"

    except (ValueError, TypeError):
        return str(value)


def find_history_column(df, candidates):
    """Find a matching training-history column."""

    lookup = {str(column).lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# ============================================================
# 5. LOAD TRAINED MODEL AND NORMALIZATION
# ============================================================

@st.cache_resource
def load_trained_model():
    """Load the trained CNN-LSTM model once per Streamlit process."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)

    expected_input = (None, EXPECTED_FRAMES, EXPECTED_FEATURES)

    if input_shape != expected_input:
        raise ValueError(
            "The model input shape does not match the dashboard.\n"
            f"Expected: {expected_input}\n"
            f"Found: {input_shape}\n"
            "Check that the model was trained with 120 features "
            "and 130 frames."
        )

    if len(output_shape) != 2 or output_shape[-1] != len(EMOTIONS):
        raise ValueError(
            "The model must output eight emotion scores. "
            f"Found output shape: {output_shape}"
        )

    return model


@st.cache_data
def load_normalization():
    """Load training-only feature normalization statistics."""

    if not NORMALIZATION_PATH.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {NORMALIZATION_PATH}"
        )

    with np.load(NORMALIZATION_PATH) as data:
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)

    expected_shape = (1, 1, EXPECTED_FEATURES)

    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(
            "Normalization arrays have unexpected shapes.\n"
            f"Expected mean/std: {expected_shape}\n"
            f"Found mean: {mean.shape}, std: {std.shape}"
        )

    std = np.where(std < 1e-6, 1.0, std)

    return mean, std


@st.cache_data
def load_dashboard_data():
    """Load metrics, training history, confusion matrix, and report."""

    metrics = read_json(METRICS_PATH, {})

    history = (
        pd.read_csv(HISTORY_PATH)
        if HISTORY_PATH.exists()
        else pd.DataFrame()
    )

    confusion = (
        pd.read_csv(CONFUSION_PATH, index_col=0)
        if CONFUSION_PATH.exists()
        else pd.DataFrame()
    )

    report = read_json(REPORT_PATH, {})

    split_info = read_json(SPLIT_PATH, {})

    return metrics, history, confusion, report, split_info


# ============================================================
# 6. SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 24px 0;">
            <div style="font-size: 2rem;">🎙️</div>
            <div style="
                font-size: 1.55rem;
                font-weight: 800;
                color: #F8FAFC;
            ">
                VoxSense <span style="color:#A78BFA;">AI</span>
            </div>
            <div style="
                color:#94A3B8;
                font-size:0.78rem;
                margin-top:5px;
            ">
                Speech Emotion Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Workspace</div>',
        unsafe_allow_html=True,
    )

    PAGES = [
        "Overview",
        "Analyze Audio",
        "Audio Visualization",
        "Model Performance",
        "Prediction History",
        "About",
    ]

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Overview"

    if "page_navigation" in st.session_state:
        st.session_state.selected_page = st.session_state.pop(
            "page_navigation"
        )

    page = st.radio(
        "Navigation",
        PAGES,
        key="selected_page",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div class="glass-card">
            <div style="font-weight:700;color:#E2E8F0;">
                Model information
            </div>
            <p class="muted">
                CNN-LSTM<br>
                8 emotion classes<br>
                130 frames × 120 features
            </p>
            <span class="status-pill">Local model</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="small-note">
            Academic project prototype.<br>
            Predictions are experimental and should not be used
            as definitive evidence of a person's emotional state.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 8. LOAD DASHBOARD ARTIFACTS
# ============================================================

metrics, history_df, confusion_df, report, split_info = (
    load_dashboard_data()
)

test_accuracy = get_metric_value(
    metrics,
    [
        "test_accuracy",
        "test_acc",
        "accuracy",
    ],
)

test_macro_f1 = get_metric_value(
    metrics,
    [
        "test_macro_f1",
        "macro_f1",
        "test_f1_macro",
    ],
)

test_recordings = get_metric_value(
    metrics,
    [
        "test_samples",
        "test_recordings",
        "n_test",
    ],
    default=300,
)

epochs_completed = (
    len(history_df)
    if not history_df.empty
    else get_metric_value(
        metrics,
        ["epochs_completed", "epochs"],
        default="N/A",
    )
)


# ============================================================
# 9. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    show_hero(
        "Understand the emotion behind the voice.",
        (
            "An audio intelligence dashboard powered by a "
            "Convolutional Neural Network and Long Short-Term "
            "Memory network. Upload a speech recording, inspect "
            "its acoustic features, and explore the model's "
            "predicted emotion distribution."
        ),
    )

    st.markdown("### System overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric(
            "Test accuracy",
            format_percentage(test_accuracy),
            "Actor-independent evaluation",
        )

    with col2:
        show_metric(
            "Macro F1",
            format_percentage(test_macro_f1),
            "Performance across emotion classes",
        )

    with col3:
        show_metric(
            "Test recordings",
            str(test_recordings),
            "Held-out actor recordings",
        )

    with col4:
        show_metric(
            "Training epochs",
            str(epochs_completed),
            "Recorded training history",
        )

    st.markdown("### Explore the platform")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:1.6rem;">🎧</div>
                <h3>Audio intelligence</h3>
                <p class="muted">
                    Upload a WAV recording and run inference
                    through the trained CNN-LSTM model.
                    Review predicted emotion scores and export
                    the results.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open audio analysis",
            key="overview_analyze",
            use_container_width=True,
        ):
            st.session_state.page_navigation = "Analyze Audio"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:1.6rem;">📊</div>
                <h3>Model diagnostics</h3>
                <p class="muted">
                    Explore training and validation curves,
                    the confusion matrix, and the classification
                    report generated during model evaluation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "View model performance",
            key="overview_performance",
            use_container_width=True,
        ):
            st.session_state.page_navigation = "Model Performance"
            st.rerun()

    st.markdown("### Emotion categories")

    emotion_cols = st.columns(4)

    for index, emotion in enumerate(EMOTIONS):
        with emotion_cols[index % 4]:
            color = EMOTION_COLORS[emotion]

            st.markdown(
                f"""
                <div style="
                    background:rgba(17,24,39,0.75);
                    border:1px solid {color}55;
                    border-radius:14px;
                    padding:16px;
                    margin-bottom:12px;
                ">
                    <div style="
                        width:9px;
                        height:9px;
                        border-radius:50%;
                        background:{color};
                        display:inline-block;
                        margin-right:8px;
                    "></div>
                    <span style="
                        font-weight:700;
                        color:#E2E8F0;
                    ">{emotion}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Model pipeline")

    pipeline_cols = st.columns(5)

    pipeline_steps = [
        ("01", "Audio input", "WAV recording"),
        ("02", "Feature extraction", "MFCC + derivatives"),
        ("03", "Normalization", "Training statistics"),
        ("04", "CNN-LSTM", "Learned patterns"),
        ("05", "Prediction", "8 emotion scores"),
    ]

    for col, (number, title, description) in zip(
        pipeline_cols,
        pipeline_steps,
    ):
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="min-height:145px;">
                    <div style="
                        color:#A78BFA;
                        font-size:0.75rem;
                        font-weight:800;
                    ">{number}</div>
                    <div style="
                        font-weight:700;
                        margin-top:12px;
                    ">{title}</div>
                    <div class="muted" style="margin-top:7px;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not MODEL_PATH.exists():
        st.warning(
            "The trained model file was not found. "
            "Add artifacts/best_model.keras to enable predictions."
        )

    if not NORMALIZATION_PATH.exists():
        st.warning(
            "Feature normalization statistics were not found. "
            "Add artifacts/feature_normalization.npz."
        )


# ============================================================
# 10. AUDIO ANALYSIS PAGE
# ============================================================

elif page == "Analyze Audio":

    show_hero(
        "Analyze a speech recording.",
        (
            "Upload a WAV file to extract acoustic features and "
            "generate an emotion prediction using your trained "
            "CNN-LSTM model."
        ),
        eyebrow="AUDIO INTELLIGENCE",
    )

    left, right = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    with left:

        st.markdown("### Upload audio")

        uploaded_file = st.file_uploader(
            "Choose a speech recording",
            type=["wav"],
            help=(
                "Upload a WAV speech recording. "
                "For best compatibility, use a standard PCM WAV file."
            ),
        )

        if uploaded_file is not None:

            audio_bytes = uploaded_file.getvalue()

            st.audio(
                audio_bytes,
                format="audio/wav",
            )

            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="section-label">Selected file</div>
                    <div style="
                        font-weight:700;
                        margin-top:8px;
                        overflow-wrap:anywhere;
                    ">
                        {uploaded_file.name}
                    </div>
                    <div class="muted" style="margin-top:8px;">
                        Size: {len(audio_bytes) / 1024:.1f} KB
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            analyze_button = st.button(
                "✨ Analyze emotion",
                key="analyze_audio_button",
                use_container_width=True,
            )

            if analyze_button:

                try:

                    if not MODEL_PATH.exists():
                        raise FileNotFoundError(
                            "artifacts/best_model.keras is missing."
                        )

                    if not NORMALIZATION_PATH.exists():
                        raise FileNotFoundError(
                            "artifacts/feature_normalization.npz is missing."
                        )

                    with st.spinner(
                        "Extracting acoustic features and running inference..."
                    ):

                        # Load the existing trained model.
                        model = load_trained_model()

                        # Load normalization learned from training data.
                        mean, std = load_normalization()

                        # Extract waveform, feature matrix, and sample rate.
                        waveform, raw_features, sample_rate = (
                            extract_features_from_audio(
                                io.BytesIO(audio_bytes)
                            )
                        )

                        waveform = np.asarray(
                            waveform,
                            dtype=np.float32,
                        ).squeeze()

                        raw_features = np.asarray(
                            raw_features,
                            dtype=np.float32,
                        )

                        if raw_features.shape != (
                            EXPECTED_FRAMES,
                            EXPECTED_FEATURES,
                        ):
                            raise ValueError(
                                "Unexpected feature shape.\n"
                                f"Expected: "
                                f"({EXPECTED_FRAMES}, {EXPECTED_FEATURES})\n"
                                f"Received: {raw_features.shape}\n"
                                "Check src/audio_features.py."
                            )

                        if waveform.size == 0:
                            raise ValueError(
                                "The uploaded audio contains no samples."
                            )

                        if not np.isfinite(raw_features).all():
                            raise ValueError(
                                "Feature extraction returned invalid values."
                            )

                        # Apply the same training-only normalization.
                        normalized_features = (
                            raw_features[np.newaxis, :, :] - mean
                        ) / std

                        normalized_features = np.nan_to_num(
                            normalized_features,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )

                        # CNN-LSTM inference.
                        predictions = model.predict(
                            normalized_features,
                            verbose=0,
                        )

                        probabilities = np.asarray(
                            predictions[0],
                            dtype=np.float64,
                        )

                        if (
                            probabilities.shape != (len(EMOTIONS),)
                            or not np.isfinite(probabilities).all()
                        ):
                            raise ValueError(
                                "The model returned invalid prediction scores."
                            )

                        predicted_index = int(
                            np.argmax(probabilities)
                        )

                        predicted_emotion = EMOTIONS[predicted_index]

                        model_score = float(
                            probabilities[predicted_index]
                        )

                        result = {
                            "filename": uploaded_file.name,
                            "emotion": predicted_emotion,
                            "model_score": model_score,
                            "probabilities": probabilities.tolist(),
                            "waveform": waveform,
                            "sample_rate": int(sample_rate),
                            "raw_features": raw_features,
                            "normalized_features": (
                                normalized_features[0]
                            ),
                            "audio_bytes": audio_bytes,
                        }

                        st.session_state.prediction_result = result

                        st.session_state.prediction_history.append(
                            {
                                "filename": uploaded_file.name,
                                "emotion": predicted_emotion,
                                "model_score": model_score,
                                "sample_rate": int(sample_rate),
                            }
                        )

                    st.success("Audio analysis completed.")

                except Exception as exc:

                    st.error(
                        f"Could not analyze this audio: {exc}"
                    )

                    st.info(
                        "Confirm that the model, normalization file, "
                        "and feature extraction function use the same "
                        "120-feature pipeline. See the terminal for "
                        "additional debugging information if needed."
                    )

    with right:

        st.markdown("### Analysis workflow")

        workflow = [
            (
                "01",
                "Audio ingestion",
                "Read the uploaded WAV recording.",
            ),
            (
                "02",
                "Acoustic features",
                "Extract MFCC, delta, and delta-delta features.",
            ),
            (
                "03",
                "Feature normalization",
                "Apply the saved training-set mean and standard deviation.",
            ),
            (
                "04",
                "CNN-LSTM inference",
                "Generate scores for eight emotion categories.",
            ),
        ]

        for number, title, description in workflow:

            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="
                        color:#A78BFA;
                        font-weight:800;
                        font-size:0.8rem;
                    ">
                        STEP {number}
                    </div>
                    <div style="
                        font-weight:700;
                        margin-top:6px;
                    ">
                        {title}
                    </div>
                    <div class="muted" style="margin-top:6px;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Display the latest prediction.
    result = st.session_state.prediction_result

    if result is not None:

        st.divider()

        st.markdown("### Prediction results")

        col1, col2 = st.columns(
            [0.9, 1.1],
            gap="large",
        )

        with col1:

            color = EMOTION_COLORS[result["emotion"]]

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="section-label">
                        PREDICTED EMOTION
                    </div>
                    <div class="prediction-emotion"
                         style="color:{color};">
                        {result["emotion"]}
                    </div>
                    <div class="muted">
                        Model output score:
                        <strong>
                            {result["model_score"] * 100:.2f}%
                        </strong>
                    </div>
                    <div class="small-note" style="margin-top:12px;">
                        This is the model's highest output score,
                        not a calibrated measure of the true probability
                        that a person is experiencing this emotion.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            probability_df = pd.DataFrame(
                {
                    "Emotion": EMOTIONS,
                    "Model score": result["probabilities"],
                }
            )

            probability_df = probability_df.sort_values(
                "Model score",
                ascending=True,
            )

            fig = px.bar(
                probability_df,
                x="Model score",
                y="Emotion",
                orientation="h",
                color="Emotion",
                color_discrete_map=EMOTION_COLORS,
                title="Emotion score distribution",
            )

            fig.update_layout(
                showlegend=False,
                xaxis=dict(
                    tickformat=".0%",
                    range=[0, 1],
                    title="Model output score",
                ),
                yaxis_title="",
            )

            fig.update_traces(
                hovertemplate=(
                    "%{y}<br>Score: %{x:.2%}<extra></extra>"
                )
            )

            st.plotly_chart(
                style_plot(fig, 370),
                use_container_width=True,
            )

        st.markdown("### Export prediction")

        export_df = pd.DataFrame(
            {
                "emotion": EMOTIONS,
                "model_score": result["probabilities"],
            }
        )

        export_df["predicted_emotion"] = result["emotion"]
        export_df["audio_filename"] = result["filename"]
        export_df["sample_rate"] = result["sample_rate"]

        st.download_button(
            label="⬇️ Download prediction scores (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="voxsense_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            f"Sample rate: {result['sample_rate']} Hz · "
            f"Feature matrix: "
            f"{result['raw_features'].shape[0]} frames × "
            f"{result['raw_features'].shape[1]} features"
        )

        st.info(
            "Emotion recognition is an uncertain inference from "
            "acoustic patterns. It cannot reliably establish a "
            "person's actual feelings, intentions, or mental state."
        )

    else:

        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:1.5rem;">🧠</div>
                <h3>Ready for inference</h3>
                <p class="muted">
                    Upload a WAV file and select
                    <strong>Analyze emotion</strong> to view
                    the model's prediction and score distribution.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 11. AUDIO VISUALIZATION PAGE
# ============================================================

elif page == "Audio Visualization":

    show_hero(
        "Explore the acoustic signal.",
        (
            "Inspect the uploaded waveform and the feature maps "
            "processed by the CNN-LSTM. Run an analysis first to "
            "populate this workspace."
        ),
        eyebrow="SIGNAL LAB",
    )

    result = st.session_state.prediction_result

    if result is None:

        st.warning(
            "No analyzed audio is available. Open Analyze Audio, "
            "upload a WAV file, and run a prediction first."
        )

        if st.button("Go to Analyze Audio"):
            st.session_state.page_navigation = "Analyze Audio"
            st.rerun()

    else:

        waveform = result["waveform"]
        sample_rate = result["sample_rate"]

        duration = len(waveform) / sample_rate

        time_axis = np.arange(len(waveform)) / sample_rate

        st.markdown("### Audio waveform")

        fig_wave = go.Figure()

        fig_wave.add_trace(
            go.Scattergl(
                x=time_axis,
                y=waveform,
                mode="lines",
                name="Amplitude",
                line=dict(
                    color="#8B5CF6",
                    width=1,
                ),
            )
        )

        fig_wave.update_layout(
            title=f"Waveform · {result['filename']}",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
        )

        st.plotly_chart(
            style_plot(fig_wave, 340),
            use_container_width=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            show_metric(
                "Duration",
                f"{duration:.2f} s",
                "Audio recording length",
            )

        with c2:
            show_metric(
                "Sample rate",
                f"{sample_rate:,} Hz",
                "Feature extractor output",
            )

        with c3:
            show_metric(
                "Frames",
                str(result["raw_features"].shape[0]),
                "Fixed-length feature sequence",
            )

        st.divider()

        st.markdown("### Acoustic feature maps")

        feature_type = st.selectbox(
            "Select feature representation",
            [
                "MFCC",
                "Delta MFCC",
                "Delta-Delta MFCC",
                "All features",
            ],
        )

        feature_matrix = result["raw_features"]

        # Features are arranged as:
        # columns 0:40   -> MFCC
        # columns 40:80  -> delta MFCC
        # columns 80:120 -> delta-delta MFCC

        if feature_type == "MFCC":

            display_features = feature_matrix[:, :40].T

            title = "Mel-Frequency Cepstral Coefficients"

            y_title = "MFCC coefficient"

        elif feature_type == "Delta MFCC":

            display_features = feature_matrix[:, 40:80].T

            title = "First-order delta MFCC"

            y_title = "Delta coefficient"

        elif feature_type == "Delta-Delta MFCC":

            display_features = feature_matrix[:, 80:120].T

            title = "Second-order delta-delta MFCC"

            y_title = "Delta-delta coefficient"

        else:

            display_features = feature_matrix.T

            title = "Complete acoustic feature representation"

            y_title = "Feature index"

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=display_features,
                x=np.arange(EXPECTED_FRAMES),
                y=np.arange(display_features.shape[0]),
                colorscale="Viridis",
                colorbar=dict(
                    title="Value",
                ),
            )
        )

        fig_heatmap.update_layout(
            title=title,
            xaxis_title="Time frame",
            yaxis_title=y_title,
        )

        st.plotly_chart(
            style_plot(fig_heatmap, 480),
            use_container_width=True,
        )

        st.caption(
            "Heatmaps display the extracted acoustic features "
            "before normalization. Color indicates the relative "
            "numerical value of each coefficient."
        )

        st.markdown("### Feature matrix preview")

        st.dataframe(
            pd.DataFrame(
                feature_matrix,
                columns=[
                    *[f"MFCC_{i + 1}" for i in range(40)],
                    *[f"Delta_{i + 1}" for i in range(40)],
                    *[f"DeltaDelta_{i + 1}" for i in range(40)],
                ],
            ).head(15),
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download extracted feature matrix",
            data=pd.DataFrame(
                feature_matrix,
                columns=[
                    *[f"MFCC_{i + 1}" for i in range(40)],
                    *[f"Delta_{i + 1}" for i in range(40)],
                    *[f"DeltaDelta_{i + 1}" for i in range(40)],
                ],
            ).to_csv(index=False).encode("utf-8"),
            file_name="voxsense_acoustic_features.csv",
            mime="text/csv",
        )


# ============================================================
# 12. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    show_hero(
        "Evaluate the model.",
        (
            "Inspect the saved training history, classification "
            "metrics, and confusion matrix from the held-out "
            "actor test set."
        ),
        eyebrow="MODEL DIAGNOSTICS",
    )

    st.markdown("### Evaluation summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        show_metric(
            "Test accuracy",
            format_percentage(test_accuracy),
            "Overall test-set accuracy",
        )

    with c2:
        show_metric(
            "Macro F1",
            format_percentage(test_macro_f1),
            "Unweighted average across classes",
        )

    with c3:
        show_metric(
            "Training epochs",
            str(epochs_completed),
            "Rows in saved training history",
        )

    # --------------------------------------------------------
    # Training curves
    # --------------------------------------------------------

    if not history_df.empty:

        st.markdown("### Training history")

        history = history_df.copy()

        epoch_column = find_history_column(
            history,
            ["epoch", "epochs"],
        )

        if epoch_column is None:
            history["epoch_display"] = np.arange(
                1,
                len(history) + 1,
            )
            epoch_column = "epoch_display"

        accuracy_column = find_history_column(
            history,
            ["accuracy", "acc"],
        )

        val_accuracy_column = find_history_column(
            history,
            ["val_accuracy", "val_acc"],
        )

        loss_column = find_history_column(
            history,
            ["loss"],
        )

        val_loss_column = find_history_column(
            history,
            ["val_loss"],
        )

        chart1, chart2 = st.columns(2)

        with chart1:

            accuracy_fig = go.Figure()

            if accuracy_column:

                accuracy_fig.add_trace(
                    go.Scatter(
                        x=history[epoch_column],
                        y=history[accuracy_column],
                        mode="lines+markers",
                        name="Training accuracy",
                        line=dict(color="#8B5CF6", width=3),
                    )
                )

            if val_accuracy_column:

                accuracy_fig.add_trace(
                    go.Scatter(
                        x=history[epoch_column],
                        y=history[val_accuracy_column],
                        mode="lines+markers",
                        name="Validation accuracy",
                        line=dict(color="#38BDF8", width=3),
                    )
                )

            accuracy_fig.update_layout(
                title="Accuracy over epochs",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
                yaxis=dict(tickformat=".0%"),
            )

            st.plotly_chart(
                style_plot(accuracy_fig, 360),
                use_container_width=True,
            )

        with chart2:

            loss_fig = go.Figure()

            if loss_column:

                loss_fig.add_trace(
                    go.Scatter(
                        x=history[epoch_column],
                        y=history[loss_column],
                        mode="lines+markers",
                        name="Training loss",
                        line=dict(color="#F59E0B", width=3),
                    )
                )

            if val_loss_column:

                loss_fig.add_trace(
                    go.Scatter(
                        x=history[epoch_column],
                        y=history[val_loss_column],
                        mode="lines+markers",
                        name="Validation loss",
                        line=dict(color="#F87171", width=3),
                    )
                )

            loss_fig.update_layout(
                title="Loss over epochs",
                xaxis_title="Epoch",
                yaxis_title="Loss",
            )

            st.plotly_chart(
                style_plot(loss_fig, 360),
                use_container_width=True,
            )

        with st.expander("View training history table"):

            st.dataframe(
                history_df,
                use_container_width=True,
            )

            st.download_button(
                "Download training history CSV",
                data=history_df.to_csv(index=False).encode("utf-8"),
                file_name="voxsense_training_history.csv",
                mime="text/csv",
            )

    else:

        st.info(
            "training_history.csv was not found or contains no data."
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    st.divider()

    st.markdown("### Confusion matrix")

    if not confusion_df.empty:

        cm = confusion_df.copy()

        # Support matrices with a label column rather than index.
        if cm.shape[1] == len(EMOTIONS) + 1:

            first_col = cm.columns[0]

            if str(first_col).lower() in {
                "emotion",
                "label",
                "class",
                "Unnamed: 0".lower(),
            }:
                cm = cm.set_index(first_col)

        # Convert to numeric values.
        cm = cm.apply(
            pd.to_numeric,
            errors="coerce",
        )

        cm = cm.dropna(
            axis=0,
            how="all",
        ).dropna(
            axis=1,
            how="all",
        )

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm.values,
                x=[str(x) for x in cm.columns],
                y=[str(y) for y in cm.index],
                colorscale="Purples",
                text=cm.values,
                texttemplate="%{text:.0f}",
                hovertemplate=(
                    "Actual: %{y}<br>"
                    "Predicted: %{x}<br>"
                    "Count: %{z}<extra></extra>"
                ),
                colorbar=dict(title="Count"),
            )
        )

        fig_cm.update_layout(
            title="Actual vs. predicted emotion classes",
            xaxis_title="Predicted class",
            yaxis_title="Actual class",
        )

        st.plotly_chart(
            style_plot(fig_cm, 480),
            use_container_width=True,
        )

        st.caption(
            "Correct predictions appear on the diagonal. "
            "Off-diagonal values indicate misclassifications."
        )

        st.download_button(
            "Download confusion matrix CSV",
            data=confusion_df.to_csv().encode("utf-8"),
            file_name="voxsense_confusion_matrix.csv",
            mime="text/csv",
        )

    else:

        st.info(
            "confusion_matrix.csv was not found in the artifacts folder."
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    st.divider()

    st.markdown("### Classification report")

    if isinstance(report, dict) and report:

        report_rows = []

        for label, values in report.items():

            if not isinstance(values, dict):
                continue

            if not any(
                key in values
                for key in (
                    "precision",
                    "recall",
                    "f1-score",
                    "support",
                )
            ):
                continue

            report_rows.append(
                {
                    "Class": label,
                    "Precision": values.get("precision"),
                    "Recall": values.get("recall"),
                    "F1-score": values.get("f1-score"),
                    "Support": values.get("support"),
                }
            )

        if report_rows:

            report_df = pd.DataFrame(report_rows)

            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download classification report CSV",
                data=report_df.to_csv(index=False).encode("utf-8"),
                file_name="voxsense_classification_report.csv",
                mime="text/csv",
            )

        else:

            st.json(report)

    else:

        st.info(
            "classification_report.json was not found."
        )

    # --------------------------------------------------------
    # Actor split information
    # --------------------------------------------------------

    with st.expander("Evaluation setup and actor split"):

        if split_info:

            st.json(split_info)

        else:

            st.write(
                "No split_info.json was found. Check the training "
                "script for the exact actor-independent split."
            )

        st.markdown(
            """
            **Evaluation note:** An actor-independent test set helps
            evaluate whether the model generalizes to speakers that
            were excluded from training. However, repeated tuning
            against the same test set can make the reported score
            optimistic. A final independent test set is preferable
            after model selection.
            """
        )


# ============================================================
# 13. PREDICTION HISTORY PAGE
# ============================================================

elif page == "Prediction History":

    show_hero(
        "Your analysis history.",
        (
            "Review predictions generated during this Streamlit "
            "session and export the results for further analysis."
        ),
        eyebrow="SESSION ACTIVITY",
    )

    prediction_history = st.session_state.prediction_history

    if prediction_history:

        history = pd.DataFrame(prediction_history)

        history.index = np.arange(1, len(history) + 1)

        st.markdown("### Recent predictions")

        st.dataframe(
            history,
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download prediction history",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="voxsense_prediction_history.csv",
            mime="text/csv",
        )

        if st.button("Clear session history"):

            st.session_state.prediction_history = []

            st.rerun()

    else:

        st.info(
            "No predictions have been recorded in this session yet. "
            "Run an analysis from the Analyze Audio page."
        )

    st.caption(
        "This history is stored in the current Streamlit session. "
        "It is not a permanent database and may disappear when "
        "the session resets."
    )


# ============================================================
# 14. ABOUT PAGE
# ============================================================

elif page == "About":

    show_hero(
        "About VoxSense AI.",
        (
            "A speech emotion recognition prototype combining "
            "acoustic feature extraction and a CNN-LSTM deep "
            "learning architecture."
        ),
        eyebrow="PROJECT INFORMATION",
    )

    st.markdown("### Project overview")

    st.markdown(
        """
        <div class="glass-card">
            <h3>Speech Emotion Recognition using CNN-LSTM</h3>
            <p class="muted">
                VoxSense AI analyzes acoustic characteristics of
                speech and predicts one of eight emotion categories.
                The application uses a convolutional neural network
                to learn local patterns from acoustic features and
                an LSTM network to model sequential relationships.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Technology stack")

        technologies = [
            ("Python", "Core implementation"),
            ("TensorFlow / Keras", "CNN-LSTM model"),
            ("Librosa", "Audio feature extraction"),
            ("NumPy / Pandas", "Numerical processing"),
            ("Streamlit", "Interactive dashboard"),
            ("Plotly", "Interactive visualizations"),
        ]

        for technology, description in technologies:

            st.markdown(
                f"""
                <div class="glass-card" style="padding:14px 16px;">
                    <div style="font-weight:700;">
                        {technology}
                    </div>
                    <div class="muted">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:

        st.markdown("### Model design")

        st.markdown(
            """
            <div class="glass-card">
                <div style="font-weight:700;">
                    Input representation
                </div>
                <p class="muted">
                    130 time frames × 120 acoustic features.
                </p>
            </div>

            <div class="glass-card">
                <div style="font-weight:700;">
                    Feature composition
                </div>
                <p class="muted">
                    40 MFCC coefficients, 40 first-order delta
                    coefficients, and 40 second-order
                    delta-delta coefficients.
                </p>
            </div>

            <div class="glass-card">
                <div style="font-weight:700;">
                    Prediction output
                </div>
                <p class="muted">
                    Eight emotion scores, with the highest
                    model output selected as the predicted class.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Dataset")

    st.markdown(
        """
        The project uses the **Ryerson Audio-Visual Database of
        Emotional Speech and Song (RAVDESS)** for model development.

        Dataset attribution and licensing requirements should be
        checked before distributing the dataset or publishing
        a public demo. Do not upload the complete dataset to
        a public source repository.
        """
    )

    st.markdown("### Limitations")

    st.markdown(
        """
        - Performance depends on the training dataset and recording
          conditions.
        - Emotion labels in acted speech do not necessarily represent
          the speaker's real emotional state.
        - Background noise, accents, microphones, and unfamiliar
          speakers can affect predictions.
        - Model output scores are not automatically calibrated
          probabilities.
        - The application is an academic prototype, not a clinical
          or psychological assessment system.
        """
    )

    st.markdown("### Project files")

    required_files = [
        MODEL_PATH,
        NORMALIZATION_PATH,
        METRICS_PATH,
        HISTORY_PATH,
        CONFUSION_PATH,
        REPORT_PATH,
    ]

    file_status = pd.DataFrame(
        [
            {
                "Artifact": path.name,
                "Status": (
                    "Available"
                    if path.exists()
                    else "Missing"
                ),
            }
            for path in required_files
        ]
    )

    st.dataframe(
        file_status,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:12px 0 20px 0;
        color:#64748B;
        font-size:0.78rem;
        line-height:1.8;
    ">
        <strong style="color:#A78BFA;">VoxSense AI</strong>
        · Speech Emotion Recognition using CNN-LSTM<br>
        Built with Python, TensorFlow, Streamlit, and Plotly<br>
        Academic research prototype · Predictions are experimental
    </div>
    """,
    unsafe_allow_html=True,

)