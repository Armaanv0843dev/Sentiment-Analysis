import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

# -------------------------
# Load Dataset
# -------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/cjp_comments.csv")

df = load_data()

# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_model():
    with open("model/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model()

# -------------------------
# Sidebar
# -------------------------

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Sentiment Analysis", "📊 Data Visualization","About"]
)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Sentiment Analysis":

    st.title("😊 Sentiment Analysis Web App")

    st.write("Predict sentiment using a trained Machine Learning model.")

    st.divider()

    # -------------------------
    # Dataset Prediction
    # -------------------------

    st.subheader("📄 Select Sentence From Dataset")

    sentence = st.selectbox(
        "Choose a sentence",
        df["text"]
    )

    if st.button("Predict Selected Sentence", use_container_width=True):

        vector = vectorizer.transform([sentence])
        prediction = str(model.predict(vector)[0]).strip().lower()

        if prediction in ["positive", "4"]:
            st.success("😊 Sentiment : POSITIVE")

        elif prediction in ["negative", "0"]:
            st.error("😠 Sentiment : NEGATIVE")

        else:
            st.info(f"😐 Sentiment : {prediction}")


    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.divider()
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # -------------------------
    # Custom Prediction
    # -------------------------

    st.subheader("✍️ Try Your Own Sentence")

    user_input = st.text_area("Enter any sentence")

    if st.button("Predict Custom Sentence", use_container_width=True):

        if user_input.strip() == "":
            st.warning("Please enter a sentence.")

        else:

            vector = vectorizer.transform([user_input])
            prediction = str(model.predict(vector)[0]).strip().lower()

            if prediction in ["positive", "4"]:
                st.success("😊 Sentiment : POSITIVE")

            elif prediction in ["negative", "0"]:
                st.error("😠 Sentiment : NEGATIVE")

            else:
                st.info(f"😐 Sentiment : {prediction}")

# ==========================================================
# VISUALIZATION PAGE
# ==========================================================

elif page == "📊 Data Visualization":

    st.title("📊 Dataset Visualization")

    st.write("Overview of the dataset.")

    # -------------------------
    # Prepare Sentiment Labels
    # -------------------------

    temp = df.copy()

    temp["sentiment"] = temp["sentiment"].replace({
        0: "Negative",
        4: "Positive"
    })

    temp["sentiment"] = temp["sentiment"].astype(str).str.title()

    positive = len(temp[temp["sentiment"] == "Positive"])
    negative = len(temp[temp["sentiment"] == "Negative"])
    neutral = len(temp[temp["sentiment"] == "Neutral"])

    # -------------------------
    # Metrics
    # -------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(temp))
    col2.metric("Positive", positive)
    col3.metric("Neutral", neutral)
    col4.metric("Negative", negative)

    st.divider()

    # -------------------------
    # Pie Chart
    # -------------------------

    st.subheader("🥧 Sentiment Distribution")

    fig, ax = plt.subplots(figsize=(5, 5))

    temp["sentiment"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")

    st.pyplot(fig)

    st.divider()

    # -------------------------
    # Count Plot
    # -------------------------

    st.subheader("📊 Count Plot")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(
        data=temp,
        x="sentiment",
        palette="Set2",
        ax=ax
    )

    st.pyplot(fig)

    st.divider()

    # -------------------------
    # Sentence Length
    # -------------------------

    st.subheader("📏 Sentence Length Distribution")

    temp["length"] = temp["text"].astype(str).apply(len)

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=temp,
        x="length",
        bins=20,
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    st.divider()

    # -------------------------
    # Average Length
    # -------------------------

    st.subheader("📈 Average Sentence Length")

    avg = temp.groupby("sentiment")["length"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))

    avg.plot(kind="bar", ax=ax)

    ax.set_ylabel("Average Characters")

    st.pyplot(fig)

    st.divider()

    # -------------------------
    # Dataset Preview
    # -------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(temp, use_container_width=True)

else:
    st.title("📖 About")
    st.markdown("""
    ## Sentiment Analysis using Machine Learning

    ### 🎯 Project Objective
    The objective of this project is to demonstrate how Natural Language Processing (NLP) and Machine Learning can be used to analyze textual data and automatically identify the sentiment expressed in user comments or reviews.

    ---
    **Developed as a Machine Learning & NLP project using Streamlit for an interactive web interface.**

    This project is a **Machine Learning-based Sentiment Analysis Web Application** developed using **Python** and **Streamlit**. It is designed to classify the sentiment of a given text into categories such as **Positive**, **Negative**, or **Neutral**.

    The application is trained on a labeled sentiment dataset using the **TF-IDF (Term Frequency–Inverse Document Frequency)** feature extraction technique and a **Logistic Regression** classifier. Users can either select a sentence from the dataset or enter their own text to predict its sentiment instantly.

    ### ✨ Key Features
    - 🔍 Predict sentiment from a dataset sentence
    - ✍️ Analyze custom user-entered text
    - 🤖 Machine Learning-based sentiment prediction
    - 📊 Interactive data visualizations
    - 📈 Dataset statistics and sentiment distribution
    - ⚡ Fast and user-friendly Streamlit interface

    ### 🛠️ Technologies Used
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorizer
    - Logistic Regression
    - Matplotlib & Seaborn

    ### 📊 Model Workflow
    1. Load and preprocess the dataset.
    2. Convert text into numerical features using TF-IDF.
    3. Train a Logistic Regression model.
    4. Predict the sentiment of new text.
    5. Visualize dataset insights and prediction results.


    """)

