# 😊 Sentiment Analysis Web App

A simple Machine Learning-based **Sentiment Analysis Web Application** built using **Python** and **Streamlit**. This application predicts whether a given English sentence expresses a **Positive** or **Negative** sentiment using a trained ML model.

---

## 📌 Features

- 🔍 Predict sentiment of any English text
- 🤖 Machine Learning model for sentiment classification
- ⚡ Fast and interactive Streamlit interface
- 📊 Clean and user-friendly UI
- 📁 Pre-trained model included
- 💻 Easy to run locally

---

## 📂 Project Structure

```
Sentiment-Analysis/
│
├── app.py                  # Streamlit Web Application
├── train_model.py          # Model Training Script
├── file_structure.txt
├── .gitignore
│
├── data/
│   └── cjp_comments.csv    # Dataset used for training
│
├── model/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Pickle

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Armaanv0843dev/Sentiment-Analysis.git
```

### 2. Navigate to the project folder

```bash
cd Sentiment-Analysis
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

If you don't have a requirements file:

```bash
pip install streamlit pandas numpy scikit-learn
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your browser.

---

## 🧠 Machine Learning Workflow

1. Load dataset
2. Text preprocessing
3. Convert text into numerical features using CountVectorizer
4. Train the Machine Learning model
5. Save trained model using Pickle
6. Predict sentiment from user input

---

## 📊 Sentiment Classes

| Sentiment | Meaning |
|-----------|---------|
| 😊 Positive | Positive opinion or emotion |
| 😞 Negative | Negative opinion or emotion |

---

## 💬 Example

**Input**

```
This product is amazing and works perfectly.
```

**Prediction**

```
Positive 😊
```

---

**Input**

```
I am very disappointed with this service.
```

**Prediction**

```
Negative 😞
```

---

## 📷 Application Preview

Add screenshots here.

```
assets/home.png
assets/predict.png
```

---

## 📌 Future Improvements

- Add Neutral sentiment class
- Improve preprocessing
- Display prediction confidence
- Support multiple ML models
- Upload CSV for batch prediction
- Model comparison dashboard
- Deploy on Streamlit Community Cloud

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Create a Pull Request

---

## 👨‍💻 Author

**Armaan Gupta**

- GitHub: https://github.com/Armaanv0843dev

---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.

It helps others discover the project and motivates future improvements.
