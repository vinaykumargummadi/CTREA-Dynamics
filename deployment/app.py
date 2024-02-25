import streamlit as st
from libaries import pd, np
from data_preprocessing import data_preprocessing,inverse_scaling
from recommendation import get_data_Recommendation
from regression_model import get_regression
st.set_page_config(layout="wide")



path_to_dataset=r"C:\\Users\gvinaykumar\OneDrive - DXC Production\Documents\5. DS_ML\CTREA-Dynamics\data\Pipeline_Dataset.csv"
path_recm_mdl = r"C:\\Users\gvinaykumar\OneDrive - DXC Production\Documents\5. DS_ML\CTREA-Dynamics\deployment\knn_model.pkl"
path_reg_mdl= r"C:\\Users\gvinaykumar\OneDrive - DXC Production\Documents\5. DS_ML\CTREA-Dynamics\deployment\pipeline_model.pkl"

df=pd.read_csv(path_to_dataset)

Features_reg_mdl =['List Year', 'Assessed Value', 'Sales Ratio', 'Minimum Estimated Occupancy',
                   'year', 'Property Type_Condo', 'Property Type_Single Family',
                   'Reason Category_Property Change & Development']
Features_recm_mdl =['Assessed Value', 'Sales Ratio', 'Minimum Estimated Occupancy',
                       'Property Type_Condo', 'Property Type_Single Family', 
                       'Reason Category_Property Change & Development','Sale Amount']

user_Featuers= []
user_df=[]

List_year=list(df['List Year'].unique())
year=list(df['year'].unique())
county=list(df['County'].unique())
property_type=list(df['Property Type'].unique())
Reason_category=list(df['Reason Category'].unique()[:-1])
List_year.sort(reverse=True)
year.sort(reverse=True)

processed_data = data_preprocessing(df) #df should be replaced with user DF
user_data=processed_data.sample(1)
recommendation_df = user_data[Features_recm_mdl[:-1]]
# regression_df = user_data[Features_reg_mdl]

# regresssion_res=get_regression(regression_df,path_reg_mdl)
regresssion_res=1234
recommend_df=get_data_Recommendation(recommendation_df,path_recm_mdl,processed_data,regresssion_res)

recommend_df=inverse_scaling(recommend_df)



with open('deployment\style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

st.title("CRETA Dynamis - Advanced House Price Prediction Model")
with st.form("user_input"):
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c7, c8, c9 = st.columns(3)
    with c1.container():
        user_List_year = st.selectbox('Property Listing Year',(List_year),index=None,placeholder="Choose listing year")
    with c2.container():
        user_saleratio=st.number_input("Sales Ratio",step=0.5)
    with c3.container():
        user_year = st.selectbox('Property Sale Recorded Year',(year),index=None,placeholder="Choose Recorded year")
    with c4.container():
        user_county = st.selectbox('County',(county),index=None,placeholder="Choose County")  
    with c5.container():
        user_property_type = st.selectbox('Property type',(property_type),index=None,placeholder="Choose Property Type")  
    with c6.container():
        user_min_estimated=st.number_input("Minimum Estimated Occupancy",step=1)
    with c7.container():
        user_sale_ratio=st.number_input("Property Valuation in USD",step=10)
    with c8.container():
        user_saleratio=st.number_input("Street Number",step=1)
    with c9.container():
        user_reason_category = st.selectbox('Reason for Sale',(Reason_category),index=None,placeholder="Choose Reason")  
    st.write('You selected:', user_List_year)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        if user_List_year == None or user_saleratio ==0 or user_year==None or user_county == None or user_property_type == None or user_min_estimated == 0 or user_sale_ratio ==0 or user_reason_category ==None:
            st.error('Please provide input for all featuers')
        # st.write("slider", user_reason_category, "checkbox", user_sale_ratio)

st.write("Outside the form")

col1, col2 = st.columns([3,7])
# col3, col4 = st.columns([3,7])
# col3, _ = st.columns([3,7])
with col1.container(border=True):
    st.write("Regression Model")
    # st.write(regresssion_res)

# with col2.container(border=True):
#     st.write("Timeline Model")

with col2.container(border=True):
    st.write("Recommendation Model")
    st.dataframe(recommend_df, use_container_width=True)

# with col4.container(border=True):
#     st.write("Map Model")


