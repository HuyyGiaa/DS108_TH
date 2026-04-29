import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import math

def overview_dataset(df, dataset_name, target_col=None):
    print(f"Overview of {dataset_name}:")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Number of rows: {len(df)}")
    
def separate_columns(df, target_col=None):
    num_cols = df.select_dtypes(include='number').columns.to_list()
    cat_cols = df.select_dtypes(exclude='number').columns.to_list()
    
    items = [item for item in df.columns if "CD" in item or "DIV" in item or "NO" in item]
    num_cols_clean = [item for item in num_cols if "CD" not in item and "DIV" not in item and "NO" not in item]
    cat_cols_clean = list(set(cat_cols + items))
    
    # if target_col in num_cols: 
    #     num_cols.remove(target_col)

    # if target_col in cat_cols:
    #     cat_cols.remove(target_col)
    
    return num_cols_clean, cat_cols_clean

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
        if df[col].nunique() > threshold and col not in cols_to_drop:
            cols_to_drop.append(col)
    
    return df.drop(columns=cols_to_drop)
    
def hist_plot(df, num_cols):
    df_num = df[num_cols]
    df_drop = drop_useless_columns(df_num)
    
    total_plots = len(df_drop.columns)
    
    if total_plots == 0:
        print("Không có cột dữ liệu nào để vẽ.")
        return


    num_cols_grid = 4
    num_rows_grid = math.ceil(total_plots / num_cols_grid)
    
    plt.figure(figsize=(16, 4 * num_rows_grid))
    
    for i, col in enumerate(df_drop.columns):
        plt.subplot(num_rows_grid, num_cols_grid, i + 1)
        sns.histplot(data=df_drop, x=col, kde=True, color='blue')
        plt.title(f'Histogram of {col}') 
        
    plt.tight_layout() 
    plt.show()