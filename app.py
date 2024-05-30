import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_text(text):
  text = text.lower()                          #Lowercase
  text = nltk.word_tokenize(text)              #Tokenization
  y = []
  for i in text:                               #Considering only alphanumeric
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()
  for i in text :                             #Removing Stopwords and Punctuation
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text :                             #Stemming
    y.append(ps.stem(i))
    
  return " ".join(y)

st.title("Email Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    vectored_sms = tfidf.transform([transformed_sms])

    result = model.predict(vectored_sms)[0]

    if result == 1:
        st.header("Spam")
    else :
        st.header("Not spam")
