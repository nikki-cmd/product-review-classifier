import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def divide_chunks(l, n):
    
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def get_key_words(x):
    vectorizer = TfidfVectorizer(stop_words='english')
    keys = []
    for i in range(0, len(x)):
        matrix = vectorizer.fit_transform(x[i])

        feature_names = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        sorted_keywords = [word for _, word in sorted(zip(scores, feature_names), reverse=True)]
        for j in range(0, len(sorted_keywords)):
            keys.append(sorted_keywords[j])
    keys = [word for word in keys if word not in stopwords.words('english')]
    return keys

def vectorize_texts(texts, key_words):
    vectorizer = TfidfVectorizer(vocabulary=key_words, stop_words='english')
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray()

def get_data():
    print("Data in preprocessing")
    path = "datasets/Reviews.csv" 

    df = pd.read_csv(path)
    df = pd.DataFrame(df)
    
    bad_df = df[(df.Score == 1) | (df.Score == 2)]
    good_df = df[(df.Score == 3) | (df.Score == 4)]
    excellent_df = df[df.Score == 5]
    
    x_bad = list(divide_chunks(list(bad_df.Text), 1000))
    x_good = list(divide_chunks(list(good_df.Text), 1000))
    x_excellent = list(divide_chunks(list(excellent_df.Text), 1000))
            
    keys_bad = get_key_words(x_bad)
    keys_good = get_key_words(x_good)
    keys_excellent = get_key_words(x_excellent)
    
    combined_keys = list(set(keys_bad) | set(keys_good) | set(keys_excellent))
    
    # Векторизуем все тексты
    X = vectorize_texts(df['Text'], combined_keys)
    y = df['Label'].values  # Метки классов
    
    print("Preprocessing finished")
    return X, y, combined_keys


