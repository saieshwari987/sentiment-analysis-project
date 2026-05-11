import pandas as pd
import pickle
import nltk
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Stopwords set
stop_words = set(stopwords.words('english'))

# Preprocessing
def preprocess(text):
    text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# Load dataset
df = pd.read_csv("train.csv", encoding='latin-1')

# Use only small part for speed
df = df.head(5000)

# IMPORTANT: CHECK COLUMN NAMES
print(df.columns)

# -----------------------------
# DATASET COLUMN FIX
# -----------------------------

# If dataset has:
# text -> review
# sentiment -> positive/negative
# Clean text
df['clean'] = df['text'].apply(preprocess)

# Convert labels
df['sentiment'] = df['sentiment'].map({
    'Negative': 0,
    'Neutral': 1,
    'Positive': 2
})

# Remove null rows
df = df.dropna()

# Features and labels
X = df['clean']
y = df['sentiment']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(max_iter=200)

# Train
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", round(accuracy * 100, 2), "%")

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")