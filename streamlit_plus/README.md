# ⌚ Luxury Watch Price Prediction

Predict the price of **luxury watches** using machine learning.  
This project demonstrates data preprocessing, pipeline building, and an interactive Streamlit web application.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/) 
[![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange?logo=scikit-learn)](https://scikit-learn.org/) 

---

## 📌 Project Overview

This project predicts **luxury watch prices** based on features such as brand, model, case material, strap material, movement type, and more.  
The web app allows users to input watch specifications and get a **price estimate in USD**.

---

## 🛠️ Technologies Used

- **Python** – main programming language
- **Streamlit** – interactive web interface
- **scikit-learn** – pipelines, preprocessing, regression models
- **LightGBM / Random Forest / Ridge Regression** – ML estimators
- **Pandas & NumPy** – data manipulation

---

## ⚡ Features

- Clean and preprocess raw watch data
- Encode categorical variables and handle missing values
- Train regression models with pipeline structure
- Interactive price prediction via Streamlit app
- Save and load trained pipeline models for reuse

---

## 📂 Project Structure
Streamlit/
└── streamlit_plus/
├── models/
│ └── final_lgb_model.pkl
├── pipeline.py
├── streamlit_app.py
├── Luxury watch.csv
└── final_pipeline.pkl



---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/luxury-watch-prediction.git
cd luxury-watch-prediction/streamlit_plus
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the Streamlit app:
```
streamlit run streamlit_app.py
```

## 📊 Notes

This project is for educational purposes only.

Model performance may vary depending on dataset size and estimator choice.

Predictions are estimates and should not be used for real financial decisions.