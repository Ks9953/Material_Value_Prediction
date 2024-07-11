import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the model, scaler, tfidf vectorizer, and imputer
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
tfidf = joblib.load('tfidf.pkl')
imputer = joblib.load('imputer.pkl')

# Define the Streamlit app
def main():
    st.title("Material Value Prediction")

    # User inputs
    item_name = st.text_input('Item Name')
    nos = st.number_input('Nos', min_value=1, value=1)
    wt = st.number_input('Weight (Wt)', min_value=0.0)
    lot_rejection_count = st.number_input('Lot Rejection Count', min_value=0)
    lot_creation_dt = st.date_input('Lot Creation Date')
    
    # Preprocess the input data
    year = lot_creation_dt.year
    month = lot_creation_dt.month
    day = lot_creation_dt.day
    day_of_week = lot_creation_dt.weekday()
    
    # Apply TF-IDF to the input item name
    item_name_tfidf = tfidf.transform([item_name]).toarray()
    item_name_tfidf_df = pd.DataFrame(item_name_tfidf, columns=tfidf.get_feature_names_out())
    
    # Create input features
    input_data = pd.DataFrame([[nos, wt, lot_rejection_count, year, month, day, day_of_week]], 
                              columns=['Nos', 'Wt', 'Lot rejection count', 'Year', 'Month', 'Day', 'DayOfWeek'])
    input_data = pd.concat([input_data.reset_index(drop=True), item_name_tfidf_df], axis=1)
    

    # Impute missing values if any
    input_data = imputer.transform(input_data)
    
    # Standardize the input features
    input_data = scaler.transform(input_data)
    
    # Prediction
    if st.button('Predict'):
        prediction = model.predict(input_data)
        st.write(f'Predicted Material Value: {prediction[0]}')

if __name__ == '__main__':
    main()