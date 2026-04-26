import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv('/spam.csv', encoding='latin-1')

#Clean up 
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

#Transform label
df['label'] = df['label'].map({'ham': 1, 'spam': 0})

#Train/Set
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

#Result
y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred))

#Lime Function
def predict_proba(texts):
    X = vectorizer.transform(texts)
    return model.predict_proba(X)