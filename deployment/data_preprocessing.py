from libaries import pd, np, StandardScaler,SimpleImputer
scaler_obj = StandardScaler()


def data_preprocessing(df):
    scaled_df=scaling_encoding(df)
    return scaled_df


def scaling_encoding(df):
    # List of columns to one-hot encode
    columns_to_one_hot_encode = ['year','Property Type', 'County', 'Reason Category']

    # Apply one-hot encoding
    data_encoded = pd.get_dummies(df, columns=columns_to_one_hot_encode, drop_first=True)

    num_df = data_encoded[['Assessed Value', 'Sales Ratio', 'Minimum Estimated Occupancy','Sale Amount']]
    cat_df = data_encoded.drop(['Assessed Value', 'Sales Ratio', 'Minimum Estimated Occupancy','Sale Amount'],axis=1)
    #Imputing missing values
    num_imputer = SimpleImputer(strategy='mean', fill_value=0)
    num_imputed = num_imputer.fit_transform(num_df)


    #Scaling and encoding
    
    num_scaled_df=scaler_obj.fit_transform(num_imputed)
    num_scaled_df = pd.DataFrame(num_scaled_df,columns=['Assessed Value', 'Sales Ratio', 'Minimum Estimated Occupancy','Sale Amount'])

    scaled_encoded_df= pd.concat([num_scaled_df,cat_df],axis=1)
    return scaled_encoded_df

def inverse_scaling(recommend_df):
    pre_Inversescaling=recommend_df[['Assessed Value', 'Sales Ratio', 'Minimum Estimated Occupancy','Sale Amount']]
    # pre_Inverseencoding = recommend_df.drop(['Assessed Value', 'Sales Ratio', 'Minimum Estimated Occupancy','Sale Amount'])
    
    #applying Inverser Scaling and encoding
    Inverse_scaled_df=scaler_obj.inverse_transform(pre_Inversescaling)
    Inverse_scaled_df = pd.DataFrame(Inverse_scaled_df,columns=['Assessed Value', 'Sales Ratio', 'Minimum Estimated Occupancy','Sale Amount'])
    # Inverse_encoding_df=pd.from_dummies(pre_Inverseencoding)
    # Inverse_scaled_encoded_df= pd.concat([Inverse_scaled_df,Inverse_encoding_df],axis=1)
    
    return Inverse_scaled_df