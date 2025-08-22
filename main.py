import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
data = pd.read_csv('IMDB Dataset.csv')

# Display the first few rows of the dataset
print(data.head())

data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'negative' else 0)


# Display any null values in the dataset
# print(data.isnull().sum())
# print(data.isna().sum())

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess the text data
def preprocess_text(text):
    # Change it to lower case
    text = text.lower()
    # Tokenize the text
    tokens = re.findall(r'\b\w+\b',text)
    # Remove punctuation and special characters
    # tokens = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join the tokens back into a single string
    return ' '.join(tokens)

# Preprocess the 'review' column
data['review'] = data['review'].apply(preprocess_text)

# Display the first few rows of the preprocessed dataset
print(data.head())

lematizer = nltk.WordNetLemmatizer()
data['lemetized_text'] = data['review'].apply(lambda x: ' '.join([lematizer.lemmatize(word) for word in x.split()]))

# print(data['lemetized_text'], data['review'])

X = data['lemetized_text']
y = data['sentiment']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_trian_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train all the models
models = {
    'MultinomialNB': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_trian_tfidf, y_train)
    model_prediction = model.predict(X_test_tfidf)
    accuracy = print(f'{name} Accuracy: {accuracy_score(y_test, model_prediction)}')
    report = print(f'{name} Classification Report:\n{classification_report(y_test, model_prediction)}')
    confusion = print(f'{name} Confusion Matrix:\n{confusion_matrix(y_test, model_prediction)}')
    f1 = print(f'{name} F1 Score: {f1_score(y_test, model_prediction)}')

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, model_prediction), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


with open('model.pkl', 'wb') as model_file:
    pickle.dump(models['Logistic Regression'], model_file)
    

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
