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
    df = pd.read_csv(file)

    results = []
    for text in df['text']:
        clean = preprocess(str(text))
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        results.append(pred)

    df['sentiment'] = results

    # Count sentiments
    positive = results.count(2)
    neutral = results.count(1)
    negative = results.count(0)

# Pie chart
    plt.figure()

    plt.pie(
        [positive, neutral, negative],
        labels=['Positive', 'Neutral', 'Negative'],
        autopct='%1.1f%%'
    )

    plt.savefig('static/chart.png')
    plt.close()

    text_all = " ".join(df['text'].astype(str))
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get model weights
    coefficients = model.coef_[0]

    # Create word importance dictionary
    word_scores = dict(zip(feature_names, coefficients))

    sorted_words = sorted(word_scores.items(), key=lambda x: x[1])

    top_positive = [word for word, score in sorted_words[:5]]
    top_negative = [word for word, score in sorted_words[-5:]]
    return render_template('index.html',
                    chart=True,
                    top_positive=top_positive,
                    top_negative=top_negative)

if __name__ == "__main__":
    app.run(debug=True)