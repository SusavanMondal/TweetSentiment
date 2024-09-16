import streamlit as st
import joblib
import time

# Set up the Streamlit app
st.set_page_config(page_title="Sentiment Recognizer")
st.title("Twitter Sentiment Analysis")

# Load the pre-trained model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    return joblib.load('twitter_model_compressed.pkl')

vectorizer, model = load_model_and_vectorizer()

# Input text from the user
tweet = st.text_input("Enter Your Tweet:")

# Predict sentiment when the button is pressed
if st.button("Predict"):
    if tweet:  # Ensure that the tweet is not empty
        starttime = time.time()
        
        # Transform the tweet using the loaded vectorizer
        tweet_transformed = vectorizer.transform([tweet])
        
        # Predict sentiment
        prediction = model.predict(tweet_transformed)
        
        endtime = time.time()
        
        st.write("Prediction time taken: ", round(endtime - starttime, 2), 'Seconds')
        st.write("Sentiment of the tweet is: ", prediction[0])
    else:
        st.write("Please enter a tweet to analyze.")
