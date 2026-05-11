from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('stopwords')

history = []
app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    if pred == 2:
        sentiment = "Positive 😊"
    elif pred == 1:
        sentiment = "Neutral 😐"
    else:
        sentiment = "Negative 😡"
    confidence = round(max(prob)*100,2)

    # Save history (latest 5)
    history.insert(0, (text, sentiment))
    if len(history) > 5:
        history.pop()

    return render_template('index.html',
                           result=sentiment,
                           confidence=confidence,
                           history=history)

@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    try:
        df = pd.read_csv(file)

        # Check text column
        if 'text' not in df.columns:
            return "CSV must contain a 'text' column"

        results = []

        positive = 0
        negative = 0
        neutral = 0

        positive_words = []
        negative_words = []

        for text in df['text']:

            clean = preprocess(str(text))
            vec = vectorizer.transform([clean])

            pred = model.predict(vec)[0]

            results.append(pred)

            if pred == 2:
                positive += 1
                positive_words.extend(clean.split())

            elif pred == 1:
                neutral += 1

            else:
                negative += 1
                negative_words.extend(clean.split())

        from collections import Counter

        top_positive = Counter(positive_words).most_common(5)
        top_negative = Counter(negative_words).most_common(5)

        # Create chart
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive, negative, neutral]

        plt.figure(figsize=(5,5))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')

        import os
        chart_path = os.path.join('static', 'chart.png')

        plt.savefig(chart_path)
        plt.close()

        return render_template(
            'index.html',
            chart=True,
            top_positive=top_positive,
            top_negative=top_negative,
            positive=positive,
            negative=negative,
            neutral=neutral
        )

    except Exception as e:
        return f"Error: {str(e)}"

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
