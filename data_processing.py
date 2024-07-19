import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == 'csv':
        separator_option = ',' 
        data = pd.read_csv(uploaded_file, sep=separator_option)
    elif file_type == 'xlsx':
        data = pd.read_excel(uploaded_file)
    elif file_type == 'data':
        data = pd.read_csv(uploaded_file)
        
    for col in data.columns:
        try:
            data[col] = data[col].str.replace(',', '.').astype(float)
        except:
            pass

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_cols = data.select_dtypes(exclude=['int64', 'float64']).columns
    return data, numeric_cols, non_numeric_cols

def handle_missing_values(data, numeric_cols, non_numeric_cols, strategy):
    if strategy == "Delete rows":
        data_cleaned = data.dropna(axis=0)
    elif strategy == "Delete columns":
        data_cleaned = data.dropna(axis=1)
    elif strategy == "Impute with mean":
        imputer = SimpleImputer(strategy='mean')
        if len(numeric_cols) > 0:
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        data_cleaned = data
    elif strategy == "Impute with median":
        imputer = SimpleImputer(strategy='median')
        if len(numeric_cols) > 0:
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        data_cleaned = data
    elif strategy == "Impute with mode":
        imputer_num = SimpleImputer(strategy='most_frequent')
        imputer_cat = SimpleImputer(strategy='most_frequent')
        if len(numeric_cols) > 0:
            data[numeric_cols] = imputer_num.fit_transform(data[numeric_cols])
        if len(non_numeric_cols) > 0:
            data[non_numeric_cols] = imputer_cat.fit_transform(data[non_numeric_cols])
        data_cleaned = data
    elif strategy == "KNN Imputation":
        imputer = KNNImputer(n_neighbors=5)
        if len(numeric_cols) > 0:
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        data_cleaned = data
    return data_cleaned


def normalize_data(data, numeric_cols, strategy):
    if strategy == "Min-Max Normalization":
        scaler = MinMaxScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    elif strategy == "Z-score Standardization":
        scaler = StandardScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data