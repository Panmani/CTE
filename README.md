# CTE (Course Teacher Evaluation) - Sentiment Analysis

## Task
  * Positive or negative? (Direction)
  * How positive (-5, 5)? (Magnitude)
  * What is it positive about? What aspect? (Classification)

## Model (v1)
* ELMo LSTM for sentence embeddings
* DNN / SVM for sentiment classification based on the ELMo embeddings
* Workflow
```
python data.py
python sent2vec.py
python model.py
```
