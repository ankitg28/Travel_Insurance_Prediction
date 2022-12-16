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

def subheader(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
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
    age = st.text_input("Enter Age",25)
    employmenttype = st.text_input("What's your Employment Type - 0. Government/Public Sector 1. Private Sector",0)
    graduateornot  = st.text_input("Are you a graduate? Yes/No",1)
    annualincome = st.text_input("Please enter Annual Income",1)
    familymembers = st.text_input("How many family members do you have?",1)
    chronicdiseases = st.text_input("Suffering from chronic diseases??",1)
    frequentflyer = st.text_input("Do you fly often?",1)
    evertravelledabroad = st.text_input("Have you ever travelled abroad?",1)
    result=""
    #         result=predict_note_authentication(0.6,0,0,0.066667,0.571429,1,1,1)
    if st.button("Predict"):
        if int(age) <= 0:
            st.error("Please enter a valid age")
        elif not(int(employmenttype) == 0 or int(employmenttype) == 1):
            st.error("Please enter a valid Employment type - 0 or 1")
        elif not(int(graduateornot) == 0 or int(graduateornot) ==1):
            st.error("Please type 0 or 1")
        elif not(int(chronicdiseases) == 0 or int(chronicdiseases) ==1):
            st.error("Please type 0 or 1")
        elif not(int(frequentflyer) == 0 or int(frequentflyer) ==1):
            st.error("Please type 0 or 1")
        elif not(int(evertravelledabroad) == 0 or int(evertravelledabroad) ==1):
            st.error("Please type 0 or 1")
        elif int(annualincome) <= 0:
            st.error("Please Enter a valid Income")
        else:
#             return
            result = predict_note_authentication(age,employmenttype,graduateornot,annualincome,familymembers,chronicdiseases,frequentflyer,evertravelledabroad)
            if result == 0:
                st.subheader("As per inputs, Model prediction is that you will not buy Travel Insurance!!")
            else:
                st.subheader("As per inputs, Model prediction is that you will buy Travel Insurance!!")
        
#     st.success('The output is {}'.format(result))
    if st.button("About"):
        st.write("Check out my Medium article:  [link](https://medium.com/@goyalankit28/travel-insurance-prediction-journey-from-dataset-selection-to-ui-based-prediction-44eeb996f778)")
#         st.text("Medium Article - https://medium.com/@goyalankit28/travel-insurance-prediction-journey-from-dataset-selection-to-ui-based-prediction-44eeb996f778")
        st.write("Check my Python Jupyter Notebook: [link](https://github.com/ankitg28/Travel_Insurance_Prediction/blob/main/6105_Combine_Data_Cleaning_Feature_Selection_Modeling_and_Interpretability_Travel_Insurance_Prediction.ipynb)")
        st.write("Check my github repository: [link](https://github.com/ankitg28/Travel_Insurance_Prediction)")

if __name__=='__main__':
    main()
    