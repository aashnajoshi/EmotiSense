from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
import pandas as pd
import joblib

data = pd.read_csv('emoji_df.csv')
emojis = data['emoji']
labels = data['label']
data_balanced = pd.DataFrame({'emoji': emojis, 'label': labels})

neutral = data_balanced[data_balanced.label == 'neutral']
positive = data_balanced[data_balanced.label == 'positive']
negative = data_balanced[data_balanced.label == 'negative']

positive_upsampled = resample(positive, replace=True, n_samples=len(neutral), random_state=42)
negative_upsampled = resample(negative, replace=True, n_samples=len(neutral), random_state=42)
data_balanced = pd.concat([neutral, positive_upsampled, negative_upsampled])
data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(data_balanced['emoji'], data_balanced['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)
print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'emoji_classifier_model.pkl')
joblib.dump(vectorizer, 'emoji_vectorizer.pkl')