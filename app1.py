# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 02:20:31 2020

@author: Ankit Goyal
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:50:04 2022

@author: Ankit Goyal
"""


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)
print("abc")
pickle_in = open("tress_model.pkl","rb")
print("abc")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(age,employmenttype,graduateornot,annualincome,familymembers,chronicdiseases,frequentflyer,evertravelledabroad):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[age,employmenttype,graduateornot,annualincome,familymembers,chronicdiseases,frequentflyer,evertravelledabroad]])
    print(prediction)
    return prediction



def main():
#     st.title("/")
    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Travel Insurance Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.text_input("Enter Age","")
    employmenttype = st.text_input("What's your Employment Type - 1. Government/Public Sector 2. Private Sector","")
    graduateornot  = st.text_input("Are you a graduate? Yes/No","")
    annualincome = st.text_input("Please enter Annual Income","")
    familymembers = st.text_input("How many family members do you have?","")
    chronicdiseases = st.text_input("Suffering from chronic diseases??","")
    frequentflyer = st.text_input("Do you fly often?","")
    evertravelledabroad = st.text_input("Have you ever travelled abroad?","")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(age,employmenttype,graduateornot,annualincome,familymembers,chronicdiseases,frequentflyer,evertravelledabroad)
#         result=predict_note_authentication(0.6,0,0,0.066667,0.571429,1,1,1)
    if result == 0:
        st.success("As per inputs, Model prediction is that you will not buy Travel Insurance!!")
    else:
        st.success("As per inputs, Model prediction is that you will buy Travel Insurance!!")
        
#     st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    