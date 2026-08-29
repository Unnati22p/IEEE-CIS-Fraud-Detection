import streamlit as st
import joblib
import pandas as pd
import os


# ============================================================
# IEEE-CIS FRAUD DETECTION - STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="IEEE-CIS Fraud Detection",
    page_icon="🔐",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")

FRAUD_THRESHOLD = 0.8294


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    return model, feature_columns


# ============================================================
# HEADER
# ============================================================

st.title("🔐 IEEE-CIS Fraud Detection")

st.markdown(
    """
    **Machine Learning Fraud Detection using XGBoost**

    This application uses a trained XGBoost classifier to estimate
    the probability that an online transaction is fraudulent.
    """
)

st.divider()


# ============================================================
# LOAD MODEL AND FEATURES
# ============================================================

try:

    model, feature_columns = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.code(str(e))

    st.info(
        "Make sure fraud_model.pkl and features.pkl are inside the model folder."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Information")

st.sidebar.write("**Algorithm:** XGBoost Classifier")
st.sidebar.write("**Features:**", len(feature_columns))
st.sidebar.write(
    "**Fraud Threshold:**",
    FRAUD_THRESHOLD
)

st.sidebar.divider()

st.sidebar.caption(
    "The threshold was optimized during model evaluation."
)


# ============================================================
# TRANSACTION INPUT
# ============================================================

st.subheader("Transaction Features")

st.write(
    "Enter the numerical feature values used by the trained model."
)

st.warning(
    "The model expects the same preprocessed numerical features "
    "used during training."
)


# ============================================================
# FEATURE INPUTS
# ============================================================

input_values = {}

columns = st.columns(3)

for i, feature in enumerate(feature_columns):

    with columns[i % 3]:

        input_values[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f",
            key=f"feature_{i}"
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Transaction",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # Create dataframe in exact training feature order
        input_data = {
            feature: input_values[feature]
            for feature in feature_columns
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        # Generate fraud probability
        fraud_probability = float(
            model.predict_proba(input_df)[0][1]
        )

        # Apply optimized threshold
        prediction = int(
            fraud_probability >= FRAUD_THRESHOLD
        )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.2f}%"
            )

        with result_col2:

            st.metric(
                "Classification Threshold",
                f"{FRAUD_THRESHOLD * 100:.2f}%"
            )


        st.progress(
            min(max(fraud_probability, 0.0), 1.0)
        )


        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION"
            )

            st.write(
                f"The predicted fraud probability is "
                f"**{fraud_probability * 100:.2f}%**, "
                f"which is above the optimized threshold of "
                f"**{FRAUD_THRESHOLD * 100:.2f}%**."
            )

        else:

            st.success(
                "✅ NOT FRAUDULENT"
            )

            st.write(
                f"The predicted fraud probability is "
                f"**{fraud_probability * 100:.2f}%**, "
                f"which is below the optimized threshold of "
                f"**{FRAUD_THRESHOLD * 100:.2f}%**."
            )


        # ====================================================
        # TECHNICAL RESULT
        # ====================================================

        with st.expander("View Prediction Details"):

            st.write(
                {
                    "prediction": prediction,
                    "prediction_label": (
                        "Fraud"
                        if prediction == 1
                        else "Not Fraud"
                    ),
                    "fraud_probability": round(
                        fraud_probability,
                        6
                    ),
                    "threshold": FRAUD_THRESHOLD,
                    "features_used": len(feature_columns)
                }
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "IEEE-CIS Fraud Detection | XGBoost | Machine Learning Project"
)
