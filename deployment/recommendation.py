import joblib
import pickle
import pandas as pd

def get_data_Recommendation(user_data,path,processed_data,regression_value):
    path = r'C:\\Users\gvinaykumar\OneDrive - DXC Production\Documents\5. DS_ML\CTREA-Dynamics\deployment\knn_model.pkl'
    loaded_model = joblib.load(path)
    # loaded_model = pickle.load(open(path,'rb'))
    print(type(loaded_model))
    distances, indices = loaded_model.kneighbors(user_data, n_neighbors=5)
    # Extract feature names
    # feature_names = loaded_model.feature_names_
    
    # Convert indices to feature names
    # feature_names_neighbors = [feature_names[idx] for idx in indices.flatten()]
    # print("Feature Names of Neighbors:", feature_names_neighbors)

    res=processed_data.iloc[indices.flatten()]
    print(processed_data.columns)
    res=res[['Assessed Value', 'Sales Ratio', 'Minimum Estimated Occupancy','Property Type_Condo', 'Property Type_Single Family', 'Reason Category_Property Change & Development','Sale Amount']]
    res=res.sort_values("Sale Amount")
    res = compare_and_return_rows(res,regression_value)
    return res
    # return processed_data.head()

def compare_and_return_rows(df, value):
    # Step 1: Sort df by the values of the "Sale Amount" column in ascending order
    df_sorted = df.sort_values(by='Sale Amount')
    
    # Step 2: Compare the given value with the values of the "Sale Amount" column
    exact_matches = df_sorted[df_sorted['Sale Amount'] == value]
    
    # Step 3: Calculate the number of exact match rows
    rn = len(exact_matches)
    
    # Step 4: Check exceptions and return rows accordingly
    if rn == 0:
        # If no exact match, find the top 5 closest rows with higher values of "Sale Amount"
        closest_higher_rows = df_sorted[df_sorted['Sale Amount'] > value].head(5)
        if len(closest_higher_rows) < 5:
            remaining_rows_needed = 5 - len(closest_higher_rows)
            closest_lower_rows = df_sorted[df_sorted['Sale Amount'] < value].tail(remaining_rows_needed)
            return pd.concat([closest_higher_rows, closest_lower_rows])
        else:
            return closest_higher_rows
    else:
        # If there are multiple exact matches or a single exact match, get the matched rows and top closest higher rows
        matched_rows = exact_matches
        remaining_rows_needed = 5 - len(matched_rows)
        closest_higher_rows = df_sorted[df_sorted['Sale Amount'] > value].head(remaining_rows_needed)
        
        # Check for exceptions
        closest_row_index = df_sorted.index.get_loc(matched_rows.index[0])
        if closest_row_index >= len(df_sorted) - 4:
            rows_from_bottom = len(df_sorted) - closest_row_index
            return df_sorted.tail(rows_from_bottom).append(matched_rows).append(df_sorted.head(5 - rows_from_bottom))
        else:
            return pd.concat([matched_rows, closest_higher_rows])