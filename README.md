# 😊 Sentiment Analysis Web App

A simple Machine Learning-based **Sentiment Analysis Web Application** built using **Python** and **Streamlit**. This application predicts whether a given English sentence expresses a **Positive** or **Negative** sentiment using a trained ML model.

---

## 📌 Features ##

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
├── app.py
├── train_model.py
├── file_structure.txt
├── .gitignore
│
├── data/
│   └── cjp_comments.csv
|   └── datasets.csv(a twitter 1.6 million tweets datasets #for further details scroll down)
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

# 📊 Dataset

This project was trained using the **Sentiment140 Dataset**, a widely used benchmark dataset for sentiment analysis.

### Dataset Details

- **Dataset Name:** Sentiment140
- **Source:** Kaggle
- **Original Data:** 1.6 Million Twitter Tweets
- **Training File Used:** `data/datasets.csv`
- **Task:** Binary Sentiment Classification (Positive / Negative)

The original dataset was downloaded from Kaggle and renamed to **`datasets.csv`** for training purposes.

Since the dataset size is approximately **227 MB**, it exceeds GitHub's maximum file size limit of **100 MB**, so it is **not included** in this repository.

However, the repository already contains the trained model (`sentiment_model.pkl`) and vectorizer (`vectorizer.pkl`), allowing the application to run without retraining.

### Download the Dataset

Download the original dataset from Kaggle:

**https://www.kaggle.com/datasets/kazanova/sentiment140**

After downloading, place the dataset inside the `data` folder and rename it to:

```
data/datasets.csv
```

Alternatively, you can update the dataset path in `train_model.py`.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Armaanv0843dev/Sentiment-Analysis.git
```

### 2. Move into the project folder

```bash
cd Sentiment-Analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is unavailable:

```bash
pip install streamlit pandas numpy scikit-learn
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will automatically launch in your default browser.

---

## 🧠 Machine Learning Workflow

1. Load the Sentiment140 dataset
2. Perform text preprocessing
3. Convert text into numerical features using CountVectorizer
4. Train the sentiment classification model
5. Save the trained model and vectorizer using Pickle
6. Predict sentiment for user input through the Streamlit interface

---

## 📊 Sentiment Classes

| Sentiment | Description |
|-----------|-------------|
| 😊 Positive | Positive opinion or emotion |
| 😞 Negative | Negative opinion or emotion |
| 🤐 Neutral  | Neutral opinion or emotion  |

---

## 💬 Example

### Input

```
This product is amazing and works perfectly.
```

### Prediction

```
Positive 😊
```

### Input

```
I am very disappointed with this service.
```

### Prediction

```
Negative 😞
```

---

## 📷 Application Preview

Add screenshots of your application here.

```
assets/home.png
assets/predict.png
```

---

## 🚀 Future Improvements

- Add Neutral sentiment classification
- Improve text preprocessing
- Display prediction confidence score
- Compare multiple ML algorithms
- CSV batch prediction
- Deploy on Streamlit Community Cloud

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 👨‍💻 Author

**Armaan Gupta**

- GitHub: https://github.com/Armaanv0843dev

---

## ⭐ Support

If you found this project helpful, please consider giving it a **⭐ Star** on GitHub. It helps support the project and encourages future improvements.
