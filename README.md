# Arabic Dialect Classification
## Introduction
1. Using tfidf vectorizer, Logistic Regression, and other Machine Learning models aside with preprocessing techniques used for Arabic Language Tweets to classify Arabic Dialect
from text.
2. Using [AraBERT](https://github.com/aub-mind/arabert) model version 2 for Deep Learning approach and comparing results with Machine Learning Approach using Confussion Matrix, F1-score.

for more info about the repo please check [pdf slides](https://github.com/hazemhosny/ArabicDialectClassification/blob/main/ArabicDialectSentimenalAnalysis.pdf), and check [Models Directory for results](https://github.com/hazemhosny/ArabicDialectClassification/tree/main/Models)

## Deployment
#### 1. Logistic Regression Model
<img src="https://github.com/hazemhosny/ArabicDialectClassification/blob/main/POST_Example.png" alt="BotExample" width="850"/>

#### 2. AraBERTv2 Model
<img src="https://github.com/hazemhosny/ArabicDialectClassification/blob/main/POST_Example2.png" alt="BotExample" width="850"/>

## Run FastAPI Server
1. First need to download related packages in conda envirnoment
<pre>
PyTorch
Pandas
matplotlib
scikit-learn
transformers
pyarabic
emoji
nltk
</pre>

2. Make sure you activate env where all packages are downloaded.
3. Run `ModelTraining_ML.ipynb`, `ModelPrediction-AraBert.ipynb` to get models pickle files
4. After that go the saved pickle files and copy (or cut) paste to static folder for the FastAPI server within folders for ML models, and other for AraBERT model.
```bash
static
│   ├───ML_models
│   └───output_dir
```
5. Run `python main.py`
6. After running your server go to `localhost:5000/docs` in browser, and try out different POST methods with different text
7. **Enjoy your server app**

