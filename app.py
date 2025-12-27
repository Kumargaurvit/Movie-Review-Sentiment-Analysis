import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

model = load_model('Models/model.h5')

word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review)->str:
    return ' '.join([reversed_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(review: str)->str:
    words = review.lower()
    words = words.split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review: str)->str:
    review = preprocess_text(review)

    prediction = model.predict(review)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment

st.title('IMDB Movie Review Sentiment Analysis')

review = st.text_area('Enter the Movie Review to classify it as Positive or Negative','')

if st.button('Classify'):
    if len(review) == 0:
        st.write('Please enter a Movie Review!')
    sentiment = predict_sentiment(review)
    st.write(f'Movie Review Sentiment Prediction : {sentiment}')