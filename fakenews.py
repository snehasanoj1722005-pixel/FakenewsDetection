import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    "text": [
        "Government announces new policy",
        "Scientists discover new medicine",
        "Breaking: Free money for everyone",
        "Shocking news: aliens landed in city"
    ],
    "label": [1, 1, 0, 0]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# Train model
model = LogisticRegression()
model.fit(X, df["label"])

# Take user input
user_input = input("Enter news: ")

# Transform input
input_vector = vectorizer.transform([user_input])

# Predict
prediction = model.predict(input_vector)

# Output
if prediction[0] == 1:
    print("Real News")
else:
    print("Fake News")