import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

def overview_dataset(df, dataset_name, target_col=None):
    print(f"Overview of {dataset_name}:")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Number of rows: {len(df)}")
    
def separate_columns(df, target_col=None):
    num_cols = df.select_dtypes(include='number').columns.to_list()
    cat_cols = df.select_dtypes(exclude='number').columns.to_list()
    
    # if target_col in num_cols: 
    #     num_cols.remove(target_col)

    # if target_col in cat_cols:
    #     cat_cols.remove(target_col)
    
    return num_cols, cat_cols

def analyze_univariate(df, target_col=None):
    
    num_cols, cat_cols = separate_columns(df, target_col)
    
    display(Markdown("### Analyze Univariate"))
    stats_df = df[num_cols].describe().T
    
    stats_df['skewness'] = df[num_cols].skew()
    stats_df['kurtosis'] = df[num_cols].kurtosis()

    display(stats_df.round(2))

def drop_useless_columns(df):
    cols_to_drop = []
    
    # Drop columns with only one unique value
    for col in df.columns:
        if df[col].nunique() <= 1:
            cols_to_drop.append(col)
    
    threshold = len(df) * 0.9
    
    # Drop ID columns 
    for col in df.columns:
        if df[col].unique() > threshold and col not in cols_to_drop:
            cols_to_drop.append(col)
    
    return df.drop(columns=cols_to_drop)
    
    
    
