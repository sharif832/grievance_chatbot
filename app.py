import streamlit as st
import pickle

# Load model and vectorizer
with open(r"C:\Users\shari\OneDrive\Desktop\grievance_bot\model.pkl", "rb") as model_file:

    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit UI
st.title("Grievance Classifier Chatbot")
st.write("Type your grievance, and I'll tell you the concerned department!")

user_input = st.text_input("Enter your grievance:")

if st.button("Predict"):
    if user_input:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        st.success(f"This issue should be handled by: **{prediction}**")
    else:
        st.warning("Please enter a grievance.")
