import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def divide_chunks(l, n):
    
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def get_data():
    path = "datasets/Reviews.csv" 

    df = pd.read_csv(path)
    df = pd.DataFrame(df)

    print(df.head(10))
    print(df.columns)

    vectorizer = TfidfVectorizer()
    messages = list(df.Text)

    x = list(divide_chunks(messages, 1000))
            
    keys = []
    for i in range(0, len(x)):
        matrix = vectorizer.fit_transform(x[i])

        feature_names = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        sorted_keywords = [word for _, word in sorted(zip(scores, feature_names), reverse=True)]
        for j in range(0, len(sorted_keywords)):
            keys.append(sorted_keywords[j])
            
    return keys
