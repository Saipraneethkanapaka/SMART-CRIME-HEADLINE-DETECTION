import joblib

model = joblib.load("crime_headline_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
import streamlit as st

st.set_page_config(
    page_title="Smart Crime Headline Detection",
    page_icon="",
    layout="centered"
)

st.markdown("""
<style>

/* ---------------- Background ---------------- */

.stApp{
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #f8fbff 35%,
        #eef6ff 70%,
        #dfefff 100%
    );
}

/* Hide Streamlit header/footer */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ---------------- Title ---------------- */

.main-title{
    text-align:center;
    margin-top:20px;
    margin-bottom:8px;
}

.logo{
    font-size:48px;
}

.title-text{
    color:#0F172A;
    font-size:42px;
    font-weight:800;
    letter-spacing:1px;
    font-family:'Segoe UI',sans-serif;
}

/* ---------------- Subtitle ---------------- */

.subtitle{
    text-align:center;
    color:#64748B;
    font-size:18px;
    margin-bottom:35px;
}

/* ---------------- Label ---------------- */

label{
    color:#1E293B !important;
    font-size:18px !important;
    font-weight:600 !important;
}

/* ---------------- Text Area ---------------- */

textarea{

    background:rgba(255,255,255,.75) !important;

    backdrop-filter:blur(10px);
    -webkit-backdrop-filter:blur(10px);

    color:#111827 !important;

    border:2px solid #C7DFFF !important;

    border-radius:16px !important;

    padding:16px !important;

    font-size:17px !important;

    box-shadow:0 10px 25px rgba(37,99,235,.08);

    transition:all .3s ease;
}

textarea:focus{

    background:white !important;

    border:2px solid #2563EB !important;

    box-shadow:0 0 0 5px rgba(37,99,235,.15) !important;
}

textarea::placeholder{
    color:#94A3B8 !important;
}

/* ---------------- Button ---------------- */

div.stButton > button{

    width:100%;

    height:54px;

    border:none;

    border-radius:14px;

    background:linear-gradient(90deg,#2563EB,#3B82F6);

    color:white;

    font-size:18px;

    font-weight:700;

    transition:.3s;
}

div.stButton > button:hover{

    background:linear-gradient(90deg,#1D4ED8,#2563EB);

    transform:translateY(-2px);

    box-shadow:0 8px 20px rgba(37,99,235,.25);
}

/* Success & Warning */

div[data-testid="stSuccess"],
div[data-testid="stWarning"]{
    border-radius:12px;
}

/* ---------- Metric Cards ---------- */

/* Metric Card */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.95);
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #C7DFFF;
}

/* Metric Label */
div[data-testid="stMetric"] label {
    color: #1E293B !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* Metric Value */
div[data-testid="stMetric"] div {
    color: #2563EB !important;
}

/* Subheaders */
h3 {
    color: #1E293B !important;
    font-weight: 700 !important;
}

/* Main headers */
h1, h2 {
    color: #0F172A !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- Heading ----------

st.markdown("""
<div class="main-title">
<span class="logo">🛡️</span>
<span class="title-text">SMART CRIME HEADLINE DETECTION</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
AI-powered classification of crime-related news headlines
</div>
""", unsafe_allow_html=True)

# ---------- Input ----------

headline = st.text_area(
    "📝 Enter Crime Headline",
    height=190,
    placeholder="Type or paste a crime news headline here..."
)

st.write("")

# ---------- Prediction ----------

# -------------------------------
# Prediction
# -------------------------------

if st.button("🚀 Predict"):

    if headline.strip() == "":
        st.warning("⚠ Please enter a news headline.")

    else:

        # Word & Character Count
        words = len(headline.split())
        characters = len(headline)

        # Convert headline into TF-IDF features
        transformed_text = vectorizer.transform([headline])

        # Prediction
        prediction = model.predict(transformed_text)[0]

        # Probability
        probability = model.predict_proba(transformed_text)[0]

        # Probabilities
        crime_prob = probability[1] * 100
        not_crime_prob = probability[0] * 100

        st.divider()

        # -----------------------------------
        # Headline
        # -----------------------------------

        st.subheader("📰 Your Headline")
        st.info(headline)

        # -----------------------------------
        # Statistics
        # -----------------------------------

        st.subheader("📊 Headline Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Words", words)

        with col2:
            st.metric("Characters", characters)

        # -----------------------------------
        # Prediction
        # -----------------------------------

        st.subheader("🤖 Prediction")

        if prediction == 1:
            st.error("🚨 Crime Headline Detected")
        else:
            st.success("✅ Not a Crime Headline")

        # -----------------------------------
        # Confidence
        # -----------------------------------

        st.subheader("🎯 Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🚨 Crime", f"{crime_prob:.2f}%")
            st.progress(int(crime_prob))

        with col2:
            st.metric("✅ Not Crime", f"{not_crime_prob:.2f}%")
            st.progress(int(not_crime_prob))

        # -----------------------------------
        # Risk Assessment
        # -----------------------------------

        st.subheader("🚦 Risk Assessment")

        if crime_prob >= 90:
            st.error("🔴 Very High Crime Probability")

        elif crime_prob >= 70:
            st.warning("🟠 High Crime Probability")

        elif crime_prob >= 50:
            st.warning("🟡 Moderate Crime Probability")

        elif crime_prob >= 30:
            st.info("🔵 Low Crime Probability")

        else:
            st.success("🟢 Very Low Crime Probability")
            
            
st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    padding:20px;
    background:rgba(255,255,255,0.85);
    border-radius:15px;
    border:1px solid #C7DFFF;
    box-shadow:0 5px 15px rgba(37,99,235,0.10);
">

<h3 style="
    color:#2563EB;
    margin-bottom:10px;
">
🛡️ Smart Crime Headline Detection
</h3>

<p style="
    color:#475569;
    font-size:15px;
    margin:5px;
">
AI-powered Crime vs Non-Crime News Classification
</p>

<p style="
    color:#64748B;
    font-size:14px;
    margin:5px;
">
Built with ❤️ using
<b>Python</b> •
<b>Streamlit</b> •
<b>Scikit-learn</b> •
<b>TF-IDF</b> •
<b>Machine Learning Models</b>
</p>

</div>
""", unsafe_allow_html=True)