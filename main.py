from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from transformers import pipeline
from utils import *
import numpy as np

app = FastAPI(debug=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# initialize pipline
pipe = pipeline("sentiment-analysis", model="static/output_dir", device=0, return_all_scores=True)

# ML model
vectorizer = unpickle_file("static/ML_models/tfidf_vectorizer.pkl")
LogisticReg = unpickle_file("static/ML_models/LogisticReg.pkl")

@app.get("/")
def home():
    return("Arabic Dialect Project Using Arabert model")

@app.post("/Predict_ml", summary="Arabic Tweet Dialect Prediction (Logistic Regression Model)")
def predict(tweet_paragraph: str):
    text = clean_text(tweet_paragraph)
    text = np.array([text]).astype('U')
    text_tfidf = vectorizer.transform(text)
    scores = LogisticReg.predict_proba(text_tfidf)[0]
    label = list(LogisticReg.classes_)[np.argmax(scores)]
    best_label = {'label':label, 'score': scores[np.argmax(scores)]}
    return [best_label]

@app.post("/Predict", summary="Arabic Tweet Dialect Prediction (AraBERT best label)")
def predict(tweet_paragraph: str):
    result = pipe(tweet_paragraph)[0]
    max_score = 0
    best_label = {}
    for label in result:
        if label['score'] > max_score:
            max_score = label['score']
            best_label = {'label': label['label'], 'score': max_score}
    return [best_label]

@app.post("/Predict_all_labels", summary="Arabic Tweet Dialect Prediction with (AraBERT all labels)")
def predict(tweet_paragraph: str):
    result = pipe(tweet_paragraph)[0]
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port =5000, reload=True)