import streamlit as st
import joblib
import pandas as pd

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Smart Crime Headline Detection",
    page_icon="🛡️",
    layout="centered"
)

# =====================================
# LOAD MODEL FILES
# =====================================

@st.cache_resource
def load_files():
    model = joblib.load("crime_headline_model.pkl")
    vectorizer = joblib.load("count_vectorizer.pkl")
    metrics = joblib.load("model_metrics.pkl")
    best_model_name = joblib.load("best_model_name.pkl")

    return model, vectorizer, metrics, best_model_name


try:
    model, vectorizer, metrics, best_model_name = load_files()

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# =====================================
# VALIDATE MODEL METRICS
# =====================================

required_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

if not isinstance(metrics, pd.DataFrame):
    st.error("Model metrics must be a Pandas DataFrame.")
    st.stop()

if not all(col in metrics.columns for col in required_columns):
    st.error("Required columns are missing in model metrics.")
    st.stop()

if metrics.empty:
    st.error("Model metrics file is empty.")
    st.stop()

best_matches = metrics[
    metrics["Model"] == best_model_name
]

if best_matches.empty:
    st.error("Best model not found in model metrics.")
    st.stop()

best_row = best_matches.iloc[0]


# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

/* GLOBAL FONT AND BACKGROUND */

html, body, [class*="st-"],
[data-testid="stAppViewContainer"] {
    font-family: "Segoe UI", Arial, sans-serif !important;
}

.stApp {
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #f8fbff 35%,
        #eef6ff 70%,
        #dfefff 100%
    );
    color: #1e293b;
}

/* HIDE DEFAULT STREAMLIT ELEMENTS */

#MainMenu, footer, header {
    visibility: hidden;
}

/* HEADINGS */

h1, h2, h3, h4, h5, h6,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3 {
    color: #0f172a !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
    font-weight: 700 !important;
}

/* NORMAL TEXT */

p, li,
.stMarkdown p,
.stMarkdown li,
[data-testid="stMarkdownContainer"] {
    color: #334155;
    font-family: "Segoe UI", Arial, sans-serif !important;
}

/* MAIN TITLE */

.main-title {
    text-align: center;
    margin-top: 20px;
    margin-bottom: 8px;
}

.title-text {
    color: #0f172a !important;
    font-size: 36px;
    font-weight: 800;
    font-family: "Segoe UI", Arial, sans-serif !important;
}

.subtitle {
    text-align: center;
    color: #64748b !important;
    font-size: 17px;
    margin-bottom: 30px;
}

/* INPUT LABELS */

label,
[data-testid="stWidgetLabel"] p {
    color: #1e293b !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
}

/* TEXT AREA */

textarea, input {
    background: rgba(255, 255, 255, 0.88) !important;
    color: #111827 !important;
    border: 2px solid #c7dfff !important;
    border-radius: 14px !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
    font-size: 16px !important;
}

textarea {
    padding: 16px !important;
    box-shadow: 0 8px 22px rgba(37, 99, 235, 0.07);
}

textarea:focus, input:focus {
    border: 2px solid #2563eb !important;
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.13) !important;
}

/* BUTTONS */

div.stButton > button {
    width: 100%;
    min-height: 52px;
    border: none;
    border-radius: 14px;
    background: linear-gradient(
        90deg,
        #2563eb,
        #3b82f6
    );
    color: #ffffff !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
    font-size: 17px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #2563eb
    );
    color: #ffffff !important;
}

/* METRICS */

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.82);
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #c7dfff;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetricLabel"] {
    color: #334155 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #1d4ed8 !important;
    font-weight: 750 !important;
}

/* PROJECT INFORMATION CARD */

.project-card {
    background: rgba(255, 255, 255, 0.78);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #c7dfff;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 5px 15px rgba(37, 99, 235, 0.07);
}

.project-card h3 {
    color: #2563eb !important;
}

.project-card p {
    color: #475569 !important;
}

/* CAPTIONS */

.stCaption,
[data-testid="stCaption"],
[data-testid="stCaptionContainer"] {
    color: #64748b !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
}

/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 12px;
    font-family: "Segoe UI", Arial, sans-serif !important;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border: 1px solid #c7dfff;
    border-radius: 12px;
    overflow: hidden;
}

/* FOOTER */

.footer {
    text-align: center;
    padding: 20px;
    background: rgba(255, 255, 255, 0.78);
    border-radius: 15px;
    border: 1px solid #c7dfff;
    margin-top: 30px;
    color: #64748b !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# MAIN HEADING
# =====================================

st.markdown("""
<div class="main-title">
    <span style="font-size:45px;">🛡️</span>
    <span class="title-text">
        SMART CRIME HEADLINE DETECTION
    </span>
</div>

<div class="subtitle">
    AI-powered classification of crime-related news headlines
</div>
""", unsafe_allow_html=True)


# =====================================
# PROJECT INFORMATION
# =====================================

st.markdown("""
<div class="project-card">

<h3>📊 Machine Learning Project</h3>

<p style="font-size:17px;">
    Project Type: <b>Binary Classification</b>
</p>

<p>
    Classifies news headlines into two categories:
    <b>Crime</b> and <b>Non-Crime</b>.
</p>

<p>
    Feature Extraction: CountVectorizer
</p>

<p>
    Machine Learning Models: 6 Algorithms
</p>

</div>
""", unsafe_allow_html=True)


# =====================================
# SESSION STATE
# =====================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =====================================
# HEADLINE INPUT
# =====================================

st.subheader("📝 Enter Crime Headline")

headline = st.text_area(
    "Enter your news headline",
    height=150,
    placeholder="Type or paste a crime news headline here...",
    key="headline_input"
)

st.write("")


# =====================================
# BUTTONS
# =====================================

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button(
        "🚀 Predict",
        use_container_width=True
    )

with col2:
    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


if clear_button:
    st.session_state.prediction_history = []
    st.session_state.pop("headline_input", None)
    st.rerun()


# =====================================
# PREDICTION
# =====================================

if predict_button:

    if not headline.strip():

        st.warning("⚠️ Please enter a news headline.")

    else:

        try:

            # TRANSFORM HEADLINE

            transformed_text = vectorizer.transform([headline])

            # PREDICT CATEGORY

            prediction = model.predict(transformed_text)[0]

            # =====================================
            # CONFIDENCE SCORE
            # =====================================

            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    transformed_text
                )[0]

                predicted_index = list(model.classes_).index(
                    prediction
                )

                confidence = float(
                    probabilities[predicted_index] * 100
                )

            # =====================================
            # DISPLAY PREDICTION
            # =====================================

            st.divider()

            st.header("🔍 Prediction Result")

            if prediction == 1:

                st.error("🚨 Crime Headline Detected")
                result_text = "Crime"

            elif prediction == 0:

                st.success("✅ Not a Crime Headline")
                result_text = "Non-Crime"

            else:

                result_text = str(prediction)
                st.info(f"Prediction: {result_text}")

            # =====================================
            # CONFIDENCE DISPLAY
            # =====================================

            st.subheader("🎯 Prediction Confidence")

            if confidence is not None:

                st.metric(
                    label="Model Confidence",
                    value=f"{confidence:.2f}%"
                )

                st.caption(
                    "Probability assigned to the predicted "
                    "class by the trained model. This is not "
                    "necessarily a calibrated probability."
                )

            else:

                st.info(
                    "Probability-based confidence is unavailable "
                    "for this model."
                )

            # =====================================
            # SAVE PREDICTION HISTORY
            # =====================================

            st.session_state.prediction_history.append({

                "Headline": headline,

                "Prediction": result_text,

                "Confidence": (
                    f"{confidence:.2f}%"
                    if confidence is not None
                    else "Unavailable"
                )

            })

            # =====================================
            # MODEL COMPARISON
            # =====================================

            st.divider()

            st.header(
                "📊 Machine Learning Model Comparison"
            )

            st.write(
                "Performance comparison of six trained "
                "machine learning classification algorithms."
            )

            # =====================================
            # BEST PERFORMING MODEL
            # =====================================

            st.divider()

            st.subheader("🏆 Best Performing Model")

            st.markdown(
                f"### {best_model_name}"
            )

            st.metric(
                label="Test Accuracy",
                value=f"{best_row['Accuracy']:.2f}%"
            )

            # =====================================
            # COMPLETE MODEL PERFORMANCE TABLE
            # =====================================

            st.divider()

            st.subheader(
                "🥇 Complete Model Performance Ranking"
            )

            display_df = metrics.copy()

            display_df = display_df.sort_values(
                by="F1 Score",
                ascending=False
            ).reset_index(drop=True)

            display_df.insert(
                0,
                "Rank",
                range(1, len(display_df) + 1)
            )

            display_df = display_df.rename(
                columns={
                    "Model": "Machine Learning Model",
                    "Accuracy": "Accuracy (%)",
                    "Precision": "Precision (%)",
                    "Recall": "Recall (%)",
                    "F1 Score": "F1 Score (%)"
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            # =====================================
            # FINAL RESULT
            # =====================================

            st.success(
                f"🏆 {best_model_name} is the best-performing "
                "model based on F1 Score."
            )

            st.caption(
                "All evaluation metrics are calculated using "
                "the held-out test dataset."
            )

        except Exception as e:

            st.error(
                f"An error occurred during prediction: {e}"
            )


# =====================================
# PREDICTION HISTORY
# =====================================

# Uncomment this section if you want prediction history.

if st.session_state.prediction_history:

    st.divider()
    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="crime_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )