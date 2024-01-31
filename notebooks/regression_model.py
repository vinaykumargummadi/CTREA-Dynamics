import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.header('Machine Learning Regression Model')
list_year = st.selectbox('Property Listing Year',
                         [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
                          2015, 2016, 2017, 2018, 2019, 2020, 2021])
sale_recorded_year = st.selectbox('Property Sale Recorded Year',
                                  [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                                   2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
assessed_value = st.number_input('Property Valuation in USD', min_value=0.0, value=10.0,
                                 placeholder="Type a number...")
sales_ratio = st.number_input('Sales Ratio', min_value=0.00, max_value=5.00, placeholder="Type a number...")
property_type = st.selectbox('Property Type', ['Commercial', 'Residential', 'Vacant Land', 'Miscellaneous',
                                               'Apartments', 'Industrial', 'Public Utility', 'Condo',
                                               'Two Family', 'Single Family', 'Four Family', 'Three Family'])
street_number = st.number_input('Street Number ', min_value=0, value=0, placeholder="Type a number...")
Minimum_Estimated_Occupancy = st.number_input('Minimum Estimated Occupancy', min_value=1, max_value=16,
                                              placeholder="Type a number...")
county = st.selectbox('County', ['New Haven County', 'Windham County', 'Hartford County',
                                 'Fairfield County', 'Litchfield County', 'Middlesex County',
                                 'New London County', 'Tolland County', 'Other'])
reason_sale = st.selectbox('Reason for sale', ['Not defined', ' Foreclosure', ' Other', ' Family',
                                               ' Change in Property', ' Plottage', ' Use Assessment', ' Tax',
                                               ' In Lieu Of Foreclosure', ' Two Towns', ' A Will',
                                               ' Portion of Property', ' Part Interest', ' Government Agency',
                                               ' Charitable Group', ' Court Order', ' Rehabilitation Deferred',
                                               ' Inter Corporation', ' Money and Personal Property',
                                               ' Non Buildable Lot', ' Correcting Deed', ' Deed Date',
                                               ' CRUMBLING FOUNDATION ASSESSMENT REDUCTION', ' Bankrupcy',
                                               ' Auction', ' Love and Affection', ' Personal Property Exchange',
                                               ' Zoning', ' Easement', ' Cemetery', ' No Consideration'])

button = st.button('Predict Sale Amount')

data = {'List Year': [list_year],
        'year': [sale_recorded_year],
        'Assessed Value': [assessed_value],
        'Sales Ratio': [sales_ratio],
        'Property Type': [property_type],
        'Street Number': [street_number],
        'Minimum Estimated Occupancy': [Minimum_Estimated_Occupancy],
        'County': [county],
        'Reason Category': [reason_sale]}

st.markdown('User Inputs')
input_data = pd.DataFrame(data)
st.dataframe(input_data.T)

if button:
    # Define the drop_columns function
    def drop_columns(input_data):
        return input_data.drop(columns=columns_to_drop)

    # applying log transformation
    def apply_log_transformation(input_data):
        input_data['Assessed Value'] = np.log1p(input_data['Assessed Value'])
        return input_data

    columns_to_drop = ['Sales Ratio']
    categorical_features = ['Property Type', 'County', 'Reason Category']

    # deserialization
    reloaded_pickle = pickle.load(open("C:\\Users\\dhami\\Downloads\\Cap_Data\\Real_notebooks\\pipeline_model.pkl", 'rb'))
    # Use the pipeline for prediction
    prediction = reloaded_pickle.predict(input_data)
    prediction = float(prediction)
    # inverse of prediction for actual value
    predict_value = np.expm1(prediction)
    st.write('Predicted Sale Amount:', predict_value)
