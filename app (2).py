
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Rock vs Mine Prediction",
    page_icon="🛰️",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl is missing from the repository.")
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🛰️ Rock vs Mine Prediction System")

st.markdown("""
### Sonar-based Machine Learning Classification

This application uses Logistic Regression to classify underwater
objects as either Rock or Mine using 60 sonar signal features.
""")

st.info("Enter exactly 60 numerical sonar measurements.")

st.subheader("Input sonar data")

uploaded_file = st.file_uploader(
    "Upload a CSV containing one sonar reading (60 columns)",
    type=["csv"]
)

default_values = """
0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,
0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,
0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,
0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,
0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,
0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,
0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,
0.0052,0.0121,0.0124,0.0055
"""

input_text = st.text_area(
    "Enter 60 comma-separated feature values",
    value=default_values,
    height=180
)

if st.button("Predict Object", type="primary"):

    try:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, header=None)

            if data.shape != (1, 60):
                raise ValueError(
                    "Uploaded CSV must contain exactly one row "
                    "and 60 columns, without a header."
                )

            values = data.iloc[0].astype(float).to_numpy()

        else:
            values = np.fromstring(
                input_text.replace("\n", ","),
                sep=","
            )

            if len(values) != 60:
                raise ValueError(
                    f"Expected 60 values, received {len(values)}."
                )

        if not np.isfinite(values).all():
            raise ValueError("All 60 values must be finite numbers.")

        if np.any((values < 0) | (values > 1)):
            raise ValueError(
                "Feature values must be between 0 and 1."
            )

        input_array = values.reshape(1, -1)

        prediction = model.predict(input_array)[0]

        probabilities = model.predict_proba(input_array)[0]
        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )

        st.divider()

        if prediction == "R":
            st.success("Prediction: ROCK 🪨")
        else:
            st.error("Prediction: MINE 💣")

        confidence = class_probabilities[prediction] * 100

        st.metric(
            "Model confidence",
            f"{confidence:.2f}%"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Rock probability",
            f"{class_probabilities['R'] * 100:.2f}%"
        )

        col2.metric(
            "Mine probability",
            f"{class_probabilities['M'] * 100:.2f}%"
        )

        st.caption(
            "Probabilities are model estimates, not calibrated "
            "real-world safety probabilities."
        )

    except ValueError as error:
        st.warning(str(error))

    except Exception as error:
        st.error(f"Prediction error: {error}")

st.divider()

st.caption(
    "Developed using Python, Pandas, NumPy, Scikit-learn "
    "and Streamlit."
)
