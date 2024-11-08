import joblib
import os
import time

model = joblib.load('emoji_classifier_model.pkl')
vectorizer = joblib.load('emoji_vectorizer.pkl')

def predict_emotion(emoji, model, vectorizer):
    emoji_vectorized = vectorizer.transform([emoji])
    prediction = model.predict(emoji_vectorized)
    confidence = model.predict_proba(emoji_vectorized).max()
    return prediction[0], confidence

while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    choice = input("What do you want to try:\n1: Input an emoji\n2: Test Predefined emojis\nq: Quit: ")

    if choice == '1':
        emoji_to_predict = input("Type the emoji you want to analyze: ")
        label, confidence = predict_emotion(emoji_to_predict, model, vectorizer)
        print(f"Emoji: {emoji_to_predict}, Emotion: {label}, Confidence: {confidence:.2f}")
        time.sleep(2)

    elif choice == '2':
        predefined_emojis = ["🥳", "😖", "🌞", "😡", "😱", "🤔", "😎", "😴", "😈", "🤢"]
        for emoji in predefined_emojis:
            predicted_label, confidence = predict_emotion(emoji, model, vectorizer)
            print(f"Emoji: {emoji} | Predicted Label: {predicted_label}, Confidence: {confidence:.2f}")
        time.sleep(5)

    elif choice.lower() == 'q':
        print("Exiting...")
        break

    else: print("Invalid choice, please try again.")