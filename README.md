# EmotiSense
EmotiSense is a Python-based project that analyzes emojis and predicts their associated emotions (positive, neutral, or negative) with confidence scores. You can use this tool via a simple console interface or a web-based interface powered by Streamlit.

## Features
- **Emoji Emotion Prediction:** Predicts the sentiment of an emoji (positive, neutral, or negative).
- **Confidence Scores:** Provides confidence levels for each prediction.
- **Streamlit Web Interface:** A user-friendly web interface to interact with the model.
- **Pre-trained Model:** Ready-to-use pre-trained model to skip the training step.

## Usage
### All required libraries can be installed using a single-line command:
```bash
pip install -r requirements.txt
```

### To run the code:
#### Console-based version:
```bash
python main.py
```

#### Streamlit-based version:
```bash
streamlit run app.py
```

### To Re-train the Model (Optional):
```bash
python model_train.py
```
*Note:* Retraining the model requires the emoji_df.csv dataset (or edit the csv_name in line9 of `model_train.py`), it may take some time depending on your machine's performance and size of new dataset.

## Description about various files:
- *app.py:* Contains the Streamlit-based version of the main code for a user-friendly web interface.
- *emoji_classifier_model.pkl:* Pre-trained model for emoji classification.
- *emoji_df.csv:* Contains the dataset with emojis, names, and their corresponding emotion labels.
- *emoji_vectorizer.pkl:* Pre-trained vectorizer for emoji text feature extraction.
- *main.py:* Contains the console-based version of the emoji emotion analysis code.
- *model_train.py:* Contains code for training the emoji emotion classification model using a dataset.
- *requirements.txt:* File containing all required Python modules.