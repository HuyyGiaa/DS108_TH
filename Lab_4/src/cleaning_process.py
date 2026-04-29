import pandas as pd

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

def 
