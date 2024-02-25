import joblib


def get_regression(df,path):
    loaded_model = joblib.load(path)