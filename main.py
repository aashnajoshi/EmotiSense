from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv('emoji_df.csv')
emojis = data['emoji']
labels = data['label']

X_train, X_test, y_train, y_test = train_test_split(emojis, labels, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)

def predict_emotion(emoji):
    emoji_vectorized = vectorizer.transform([emoji])
    prediction = model.predict(emoji_vectorized)
    confidence = model.predict_proba(emoji_vectorized).max()

    return prediction[0], confidence

emoji_to_predict = input("Type the emoji you want to analyse: ")
label, confidence = predict_emotion(emoji_to_predict)
print(f"Emoji: {emoji_to_predict}, Emotion: {label}, Confidence: {confidence:.2f}")