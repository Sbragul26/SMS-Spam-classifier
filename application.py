import pandas as pd
import pickle
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data once
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter Stemmer and stopwords once
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Define the text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum() and i not in stop_words]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load the CSV file and preprocess the text data
df = pd.read_csv('C:\\Users\\sbrag\\Downloads\\sms-spam-classifier-main\\sms-spam-classifier-main\\spam.csv', encoding='ISO-8859-1')

# Drop any columns that are not needed and handle missing values
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore').dropna()

df['transformed_text'] = df['v2'].apply(transform_text)

# Verify the preprocessing
print("Sample transformed text:", df['transformed_text'].head())

# Initialize training data and labels
training_data = df['transformed_text'].tolist()
training_labels = df['v1'].apply(lambda x: 1 if x == 'spam' else 0).tolist()

# Initialize and fit the TfidfVectorizer
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(training_data)

# Initialize and fit the MultinomialNB model
model = MultinomialNB()
model.fit(X_train, training_labels)

# Save the fitted TfidfVectorizer and trained model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Testing the model with known samples
test_samples = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)",
    "Nah I don't think he goes to usf, he lives around here though",
    "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
    "Hello, how are you doing today?",
]

# Manually transform and predict test samples
for sample in test_samples:
    transformed_sample = transform_text(sample)
    vectorized_sample = tfidf.transform([transformed_sample])
    prediction = model.predict(vectorized_sample)[0]
    print(f"Sample: {sample}\nTransformed: {transformed_sample}\nPrediction: {'Spam' if prediction == 1 else 'Not Spam'}\n")

import streamlit as st

# Load the pre-trained vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input message
    transformed_sms = transform_text(input_sms)
    st.write("Transformed SMS:", transformed_sms)  # Debugging

    # 2. Vectorize the transformed message
    vector_input = tfidf.transform([transformed_sms])
    st.write("Vectorized input shape:", vector_input.shape)  # Debugging

    # 3. Predict using the loaded model
    result = model.predict(vector_input)[0]
    st.write("Prediction result:", result)  # Debugging

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
