import pandas as pd
import numpy as np

def check_same_dtypes(df1, df2):
    if (df1.dtypes == df2.dtypes).all():
        print("They are the same ")
    else:
        print("They are different about:")
        mismatch = df1.dtypes[df1.dtypes != df2.dtypes]
        for col in mismatch.index:
            dtype1 = df1[col].dtype
            dtype2 = df2[col].dtype
            print(f"Column '{col}': {dtype1} vs {dtype2}")
    
def check_columns_name(df1, df2):
    if (len(df1.columns) != len(df2.columns)):
        print("They have different number of columns")
        
    diff_df1 = set(df1.columns) - set(df2.columns)
    diff_df2 = set(df2.columns) - set(df1.columns)
    
    if diff_df1 or diff_df2:
        print("Two table have different columns")
        if diff_df1:
            print("Columns in df1 but not in df2:", diff_df1)
        if diff_df2:
            print("Columns in df2 but not in df1:", diff_df2)
            
    else:
        if (df1.columns == df2.columns).all():
            print("They have the same column names")
        else:
            print("They have the same columns but in a different order")
            for i, (col1, col2) in enumerate(zip(df1.columns, df2.columns)):
                if col1 != col2:
                    print(f"Difference at {i}: df1 is '{col1}' - df2 is '{col2}'")
        
def concat_df(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)

def overview_dataset(df, dataset_name, target_col=None):
    print(f"Overview of {dataset_name}:")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Number of rows: {len(df)}")
    print(f"Data overview of data {dataset_name}:, {df.describe(include='all')}")
    
def check_data_types(df, dataset_name):
    print(f"Data types in {dataset_name}:")
    print(df.dtypes)

def check_unique_values(df, dataset_name):
    print(f"Unique values in {dataset_name}:")
    print(df.nunique())

def handle_whitespace(df, dataset_name):
    print(f"Whitespace in {dataset_name}:")
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    return df

def drop_columns(df, dataset_name):
    print(f"Missing values in {dataset_name}:")
    print(df.isnull().sum())
    
    cols_to_drop = []
    for col in df.columns:
        missing_percentage = df[col].isnull().mean() * 100
        # drop column if missing values are more than 80% of the total rows
        if missing_percentage >= 80 and col not in cols_to_drop:
                print(f"Column '{col}' has {missing_percentage:.2f}% missing values. Consider dropping this column.")
                cols_to_drop.append(col)
                continue
        
        # drop column if it has only one unique value
        if df[col].nunique() <= 1 and col not in cols_to_drop:
                print(f"Column '{col}' has only one unique value. Consider dropping this column.")
                cols_to_drop.append(col)
                continue
            
        # drop column if it has more than 90% unique values
        if df[col].nunique() > len(df) * 0.9 and col not in cols_to_drop:
            print(f"Column '{col}' has {df[col].nunique()} unique values, which is more than 90% of the total rows. Consider dropping this column.")
            cols_to_drop.append(col)
            continue
        
    df.drop(columns=cols_to_drop, inplace=True)
    return df

def drop_duplicate_columns(df):
    duplicate_cols = df.columns[df.columns.duplicated()]
    if len(duplicate_cols) > 0:
        print(f"Duplicate columns found: {duplicate_cols}")
        df.drop(columns=duplicate_cols, inplace=True)
    else:
        print("No duplicate columns found.")
    return df

def export_data(df, file_path):
    df.to_csv(file_path, index=False)