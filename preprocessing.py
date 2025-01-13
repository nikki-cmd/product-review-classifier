import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

path = "datasets/Reviews.csv" 

df = pd.read_csv(path)
df = pd.DataFrame(df)

print(df.head(10))
print(df.columns)

comments = []

for i in range(0, len(df.Text)):
    comments.append(df.Text[i])

vectorizer = TfidfVectorizer()

matrix = vectorizer.fit_transform(comments[:1000])

feature_names = vectorizer.get_feature_names_out()
scores = matrix.toarray()[0]

sorted_keywords = [word for _, word in sorted(zip(scores, feature_names), reverse=True)]

print(f"Keywords{sorted_keywords}")