import streamlit as st
import joblib
import pandas as pd
import os


# ============================================================
# PAGE CONFIGURATION
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
DEMO_PATH = os.path.join(MODEL_DIR, "demo_transaction.csv")

FRAUD_THRESHOLD = 0.8294


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    return model, feature_columns


try:

    model, feature_columns = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.code(str(e))

    st.info(
        "Make sure fraud_model.pkl and features.pkl "
        "are inside the model folder."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🔐 IEEE-CIS Fraud Detection")

st.markdown(
    """
    ### Machine Learning Fraud Detection using XGBoost

    An end-to-end machine learning application for detecting
    potentially fraudulent online transactions using the
    IEEE-CIS Fraud Detection dataset.
    """
)

st.divider()


# ============================================================
# SIDEBAR - MODEL INFORMATION
# ============================================================

st.sidebar.title("📊 Model Information")

st.sidebar.metric(
    "Algorithm",
    "XGBoost"
)

st.sidebar.metric(
    "Features",
    len(feature_columns)
)

st.sidebar.metric(
    "Fraud Threshold",
    f"{FRAUD_THRESHOLD:.4f}"
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Model Performance**

    - ROC-AUC: **0.9441**
    - PR-AUC: **0.6563**
    - Precision: **0.7039**
    - Recall: **0.5621**
    - F1-Score: **0.6251**
    """
)

st.sidebar.divider()

st.sidebar.caption(
    "The classification threshold was optimized "
    "during model evaluation."
)


# ============================================================
# INTRODUCTION
# ============================================================

st.subheader("🧾 Transaction Analysis")

st.write(
    "Choose a transaction mode below and use the trained "
    "XGBoost model to estimate fraud probability."
)


# ============================================================
# TRANSACTION MODE
# ============================================================

mode = st.radio(
    "Select transaction mode",
    [
        "🎯 Demo Transaction",
        "🛠️ Custom Transaction"
    ],
    horizontal=True
)


# ============================================================
# INITIALIZE VARIABLES
# ============================================================

predict_button = False
input_df = None
demo_df = None


# ============================================================
# DEMO TRANSACTION
# ============================================================

if mode == "🎯 Demo Transaction":

    st.info(
        "The demo uses a sample set of preprocessed feature "
        "values. This allows you to test the deployed model "
        "without manually entering all 46 features."
    )

    st.subheader("Demo Transaction")

    # --------------------------------------------------------
    # LOAD DEMO TRANSACTION
    # --------------------------------------------------------

    try:

        demo_df = pd.read_csv(DEMO_PATH)

        # Keep exactly the features used by the trained model
        demo_df = demo_df[feature_columns]

        st.success(
            f"Demo transaction loaded successfully with "
            f"{len(feature_columns)} features."
        )

    except Exception as e:

        st.error(
            f"Could not load demo transaction: {str(e)}"
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY FEATURE COUNT
    # --------------------------------------------------------

    st.write(
        f"**{len(feature_columns)} model features loaded successfully.**"
    )


    # --------------------------------------------------------
    # VIEW DEMO FEATURES
    # --------------------------------------------------------

    with st.expander("View Demo Feature Values"):

        st.dataframe(
            demo_df.T.rename(
                columns={0: "Value"}
            ),
            use_container_width=True
        )


    # --------------------------------------------------------
    # DEMO PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔍 Analyze Demo Transaction",
        type="primary",
        use_container_width=True
    )

    # The dataframe used for prediction
    input_df = demo_df


# ============================================================
# CUSTOM TRANSACTION
# ============================================================

else:

    st.warning(
        "The trained model expects the same preprocessed "
        "numerical features used during training."
    )

    st.write(
        f"Enter values for the {len(feature_columns)} model features."
    )


    # --------------------------------------------------------
    # CREATE INPUT FIELDS
    # --------------------------------------------------------

    input_values = {}

    columns = st.columns(3)

    for i, feature in enumerate(feature_columns):

        with columns[i % 3]:

            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"custom_feature_{i}"
            )


    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [input_values],
        columns=feature_columns
    )


    # --------------------------------------------------------
    # CUSTOM PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True
    )


# ============================================================
# SINGLE PREDICTION SECTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if input_df is None:

            st.error(
                "No transaction data available for prediction."
            )

            st.stop()


        # ----------------------------------------------------
        # ENSURE CORRECT FEATURE ORDER
        # ----------------------------------------------------

        input_df = input_df[feature_columns]


        # ----------------------------------------------------
        # CONVERT FEATURES TO NUMERIC
        # ----------------------------------------------------

        input_df = input_df.astype(float)


        # ----------------------------------------------------
        # GENERATE FRAUD PROBABILITY
        # ----------------------------------------------------

        fraud_probability = float(
            model.predict_proba(input_df)[0][1]
        )


        # ----------------------------------------------------
        # APPLY OPTIMIZED THRESHOLD
        # ----------------------------------------------------

        prediction = int(
            fraud_probability >= FRAUD_THRESHOLD
        )


        # ----------------------------------------------------
        # PREDICTION LABEL
        # ----------------------------------------------------

        prediction_label = (
            "Fraud"
            if prediction == 1
            else "Not Fraud"
        )


        # ====================================================
        # RESULT SECTION
        # ====================================================

        st.divider()

        st.subheader("🔎 Prediction Result")


        # ----------------------------------------------------
        # MAIN RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "🚨 Potential Fraudulent Transaction"
            )

        else:

            st.success(
                "✅ Transaction Classified as Not Fraud"
            )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Decision Threshold",
                f"{FRAUD_THRESHOLD * 100:.2f}%"
            )


        with col3:

            st.metric(
                "Prediction",
                prediction_label
            )


        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.write("### Fraud Probability")

        st.progress(
            min(
                max(
                    fraud_probability,
                    0.0
                ),
                1.0
            )
        )


        # ====================================================
        # EXPLANATION
        # ====================================================

        if prediction == 1:

            st.write(
                f"""
                The model estimated a fraud probability of
                **{fraud_probability * 100:.2f}%**.

                This is above the optimized classification
                threshold of **{FRAUD_THRESHOLD * 100:.2f}%**,
                so the transaction is classified as **Fraud**.
                """
            )

        else:

            st.write(
                f"""
                The model estimated a fraud probability of
                **{fraud_probability * 100:.2f}%**.

                This is below the optimized classification
                threshold of **{FRAUD_THRESHOLD * 100:.2f}%**,
                so the transaction is classified as
                **Not Fraud**.
                """
            )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander(
            "📋 View Technical Prediction Details"
        ):

            result_data = {

                "Prediction": prediction,

                "Prediction Label": prediction_label,

                "Fraud Probability": round(
                    fraud_probability,
                    6
                ),

                "Fraud Threshold": FRAUD_THRESHOLD,

                "Features Used": len(feature_columns),

                "Model": "XGBoost Classifier"

            }

            st.json(result_data)


        # ====================================================
        # INPUT FEATURES
        # ====================================================

        with st.expander(
            "📊 View Input Features"
        ):

            st.dataframe(
                input_df.T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.code(str(e))


# ============================================================
# MODEL DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This application is a machine learning demonstration "
    "based on the IEEE-CIS Fraud Detection dataset. "
    "Predictions should not be treated as financial or "
    "security decisions without additional validation."
)

st.caption(
    "IEEE-CIS Fraud Detection • XGBoost • Streamlit"
)
