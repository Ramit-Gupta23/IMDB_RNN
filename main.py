import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , SimpleRNN , Dense,Dropout


word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}


model=tf.keras.models.load_model('imdb_simple_rnn_model.h5')


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)
        if index < 9997:
            encoded_review.append(index + 3)
        else:
            encoded_review.append(2)  # unknown token
    
    padded_review = sequence.pad_sequences([encoded_review], maxlen=100)
    return padded_review


def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]



##streamlit app

import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive or Negative).")

user_input  =st.text_area("Enter your movie review here:") 
if st.button("Classify Sentiment"):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Confidence Score: {prediction[0][0]:.4f}")

else:
    st.write("Please enter a review and click the button to classify its sentiment.") 
