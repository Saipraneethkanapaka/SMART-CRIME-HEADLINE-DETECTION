# 🛡️ Smart Crime Headline Detection

## 📖 Overview

Smart Crime Headline Detection is a Machine Learning and Natural Language Processing (NLP) based web application that classifies news headlines into two categories: **Crime** and **Non-Crime**.

The system uses **CountVectorizer** to convert textual headlines into numerical features and evaluates six different Machine Learning algorithms to identify the best-performing model.

The best model is selected based on **Crime Recall**, with **F1 Score** as the tie-breaker. The application is developed using **Python and Streamlit** and deployed on Streamlit Community Cloud.

## ✨ Features

- Classifies news headlines as Crime or Non-Crime.
- Text feature extraction using CountVectorizer.
- Uses unigram and bigram features for text representation.
- Compares six Machine Learning algorithms.
- Automatically selects the best-performing model.
- Prioritizes Crime Recall during model selection.
- Evaluates models using Accuracy, Precision, Recall, and F1 Score.
- Generates a confusion matrix and classification report.
- Interactive web interface using Streamlit.
- Online deployment for easy accessibility.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Text Vectorization:** CountVectorizer
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score
- **Dataset Format:** CSV

## 🤖 Machine Learning Algorithms

The project evaluates the following six Machine Learning algorithms:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest Classifier
4. Linear Support Vector Machine (Linear SVM)
5. Decision Tree Classifier
6. K-Nearest Neighbors (KNN)

The best model is selected based on the highest Crime Recall. If multiple models have the same Recall, the F1 Score is used as the tie-breaker.

## 🔍 Model Training Process

The training pipeline follows these steps:

1. **Dataset Loading:** Loads the dataset containing news headlines and their corresponding crime labels.
2. **Data Cleaning:** Removes missing values, empty headlines, duplicate records, and conflicting labels.
3. **Label Encoding:** Converts Crime into 1 and Non-Crime into 0.
4. **Dataset Splitting:** Divides the dataset into 60% training, 20% validation, and 20% testing.
5. **Text Vectorization:** Uses CountVectorizer with a maximum of 20,000 features and unigram/bigram extraction.
6. **Model Training:** Trains six different Machine Learning algorithms.
7. **Model Selection:** Selects the best model using Crime Recall and F1 Score.
8. **Final Evaluation:** Evaluates the trained models on unseen test data.
9. **Model Saving:** Saves the selected model, vectorizer, evaluation metrics, and model comparison results using Joblib and CSV files.

## 📁 Project Structure

```text
SMART-CRIME-HEADLINE-DETECTION/
│
├── app.py
├── train.py
├── dataset.csv
│
├── crime_headline_model.pkl
├── count_vectorizer.pkl
├── model_metrics.pkl
├── best_model_name.pkl
│
├── model_comparison.csv
├── confusion_matrix.csv
│
├── requirements.txt
└── README.md
```

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Saipraneethkanapaka/SMART-CRIME-HEADLINE-DETECTION.git
```

### 2. Navigate to the Project Directory

```bash
cd SMART-CRIME-HEADLINE-DETECTION
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning Models

```bash
python train.py
```

This step trains and evaluates all six algorithms, selects the best model, and saves the required model files.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

## 📊 Model Evaluation

The models are evaluated using the following performance metrics:

| Metric | Description |
|---|---|
| Accuracy | Measures overall prediction correctness |
| Precision | Measures how many predicted crime headlines are actually crime |
| Recall | Measures how many actual crime headlines are correctly identified |
| F1 Score | Harmonic mean of Precision and Recall |
| Confusion Matrix | Shows correct and incorrect classifications |

**Model Selection Strategy:**
- Primary metric: Crime Recall
- Secondary metric: F1 Score

The training script also generates model comparison results and a confusion matrix in CSV format.

## 🌐 Live Demo

Try the application online:

**[Smart Crime Headline Detection – Live Application](https://smart-crime-detection.streamlit.app/)**

## 🎯 How It Works

1. The user enters a news headline into the Streamlit application.
2. The headline is converted into numerical features using the saved CountVectorizer.
3. The extracted features are passed to the selected Machine Learning model.
4. The model predicts whether the headline belongs to the Crime or Non-Crime category.
5. The prediction result is displayed on the application.

## 👨‍💻 Author

**Sai Praneeth K**

GitHub: [Saipraneethkanapaka](https://github.com/Saipraneethkanapaka)