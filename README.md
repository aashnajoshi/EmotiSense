# EmotiSense
EmotiSense is a Python-based project that analyzes emojis and predicts their associated emotions (positive, neutral, or negative) with confidence scores.

## Features
- Emoji input for emotion analysis.
- Predicts emotional sentiment (positive, neutral, negative).
- Provides confidence scores for predictions.
- User-friendly web interface using Streamlit.

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

## Description about various files:
- *app.py:* Contains the Streamlit-based version of the main code for a user-friendly web interface.
- *emoji_df.csv:* Contains the dataset with emojis, names, and their corresponding emotion labels.
- *main.py:* Contains the console-based version of the emoji emotion analysis code.
- *requirements.txt:* File containing all required Python modules.
```