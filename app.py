import streamlit as st
import joblib

model = joblib.load('emoji_classifier_model.pkl')
vectorizer = joblib.load('emoji_vectorizer.pkl')

def predict_emotion(emoji):
    emoji_vectorized = vectorizer.transform([emoji])
    prediction = model.predict(emoji_vectorized)
    confidence = model.predict_proba(emoji_vectorized).max()
    return prediction[0], confidence

st.title("Emotisense: Emoji Sentiment Analysis")
st.write("Analyze the sentiment of an emoji using a trained model!")
emoji_to_predict = st.text_input("Type the emoji you want to analyze:", "😀")
st.write("For inspiration, you can explore emojis at [Emoji Keyboard](https://emojikeyboard.io/).")

if st.button("Analyze"):
    label, confidence = predict_emotion(emoji_to_predict)
    if label:
        st.write(f"**Emoji:** {emoji_to_predict}")
        st.write(f"**Emotion:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.write("Unable to analyze the emoji. Please try a different one.")