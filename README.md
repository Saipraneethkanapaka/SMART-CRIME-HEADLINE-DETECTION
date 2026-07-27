# 🛡️ Smart Crime Headline Detection

## 📖 Overview
Smart Crime Headline Detection is a Machine Learning and Natural Language Processing (NLP) web application that classifies news headlines as *Crime* or *Non-Crime. The application uses **TF-IDF Vectorization* for text preprocessing and a trained Machine Learning model for accurate prediction. The web interface is developed using *Streamlit* and deployed on *Streamlit Community Cloud*.

---

## ✨ Features
- Predicts whether a news headline is Crime or Non-Crime.
- User-friendly Streamlit web interface.
- Fast and accurate predictions using Machine Learning.
- NLP-based text preprocessing with TF-IDF.
- Deployed online for easy access.

---

## 🛠️ Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- TF-IDF Vectorizer

---

## 📁 Project Structure

```text
SMART-CRIME-HEADLINE-DETECTION/
│── app.py
│── train.py
│── crime_headline_model.pkl
│── tfidf_vectorizer.pkl
│── dataset.csv
│── requirements.txt
└── README.md
```


---

## 🚀 Installation

### Clone the repository

bash
git clone https://github.com/Saipraneethkanapaka/SMART-CRIME-HEADLINE-DETECTION.git


### Move to the project directory

bash
cd SMART-CRIME-HEADLINE-DETECTION


### Install the required packages

bash
pip install -r requirements.txt


### Run the application

bash
streamlit run app.py


---

## 🌐 Live Demo

*Streamlit App:*  
https://smart-crime-detection.streamlit.app/

---

## 🎯 How It Works

1. Enter a news headline.
2. The headline is preprocessed using the TF-IDF Vectorizer.
3. The trained Machine Learning model analyses the text.
4. The application predicts whether the headline is *Crime* or *Non-Crime*.
5. The prediction is displayed instantly.

---

## 👨‍💻 Author

*Sai Praneeth K*

GitHub: https://github.com/Saipraneethkanapaka