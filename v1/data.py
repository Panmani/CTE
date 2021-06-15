import pandas as pd
import os
import pickle

DATA_DIR = "../DATA"
NEUTRAL_COMMENTS = ["None", "none", "No", "no", "Not much", "no comment", "N/A", "N/A."]
DATASET_CACHE = "dataset.pkl"

def isEnglish(sentence):
    if not isinstance(sentence,str):
        return False
    try:
        sentence.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def read_comments(file_name):
    pos_data = pd.read_csv(os.path.join(DATA_DIR, file_name))[3:]

    comments = []
    for row in pos_data.iloc[:, 1]:
        if isEnglish(row) and row not in NEUTRAL_COMMENTS:
            comments.append(row)
    return comments

if __name__ == '__main__':
    pos_comments = read_comments("Good_features.csv")
    neg_comments = read_comments("Improvement.csv")
    all_comments = pos_comments + neg_comments
    labels = [1,] * len(pos_comments) + [0,] * len(neg_comments)

    with open(DATASET_CACHE, "wb") as dataset_file:
        pickle.dump([all_comments, labels], dataset_file)

    print("Extracted {} sentences".format(len(labels)))
