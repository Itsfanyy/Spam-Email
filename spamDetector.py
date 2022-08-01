import streamlit as st
pip install -U scikit-learn
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np



model = pickle.load(open('spam.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))


def main():
    st.title("Email Spam Classification Application")
    st.write("By : Fanisa Nimastiti & Nadea Putri Nur Fauzi")
    activites=["Classification","About"]
    choices=st.sidebar.selectbox("Select Activities",activites)
    if choices=="Classification":
        st.subheader("Classification")
        msg=st.text_input("Enter a text")
        if st.button("Predict"):
            print(msg)
            print(type(msg))
            data=[msg]
            print(data)
            vec=cv.transform(data).toarray()
            result=model.predict(vec)
            if result[0]==0:
                st.success("This is Not A Spam Email")
            else:
                st.error("This is A Spam Email")

main()
