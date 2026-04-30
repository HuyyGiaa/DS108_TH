import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
from scipy.stats import chi2_contingency

def calculate_skewness(df, num_cols):
    return df[num_cols].skew()

def calculate_kurtosis(df, num_cols):
    return df[num_cols].kurtosis()

def display_summary_statistics(df, num_cols):
    return df[num_cols].describe()

def plot_histogram(df, num_cols):
    total_plots = len(num_cols)
    cols = 3 
    rows = math.ceil(total_plots / cols)
    plt.figure(figsize=(5 * cols, 5 * rows))
    for i, col in enumerate(num_cols):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='blue')
        plt.title(f'Histogram of {col}')
    plt.tight_layout()
    plt.show()

def plot_boxplot(df, num_cols):
    total_plots = len(num_cols)
    cols = 3
    rows = math.ceil(total_plots / cols)
    plt.figure(figsize=(5 * cols, 5 * rows))
    for i, col in enumerate(num_cols):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(y=df[col].dropna(), color='lightblue')
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()
    
def plot_bivariate_num_target(df, num_cols, target_col='label', plot_type='box_plot'):
    total_plot = len(num_cols)
    cols = 3
    rows = math.ceil(total_plot / cols)
    
    plt.figure(figsize=(5 * cols, 5 * rows))
    for i, col in enumerate(num_cols):
        plt.subplot(rows, cols, i + 1)
        if plot_type == 'box_plot':
            sns.boxplot(data=df, x=target_col, y=col, hue=target_col, legend=False, palette='Set2')
            plt.title(f'Boxplot of {col} by {target_col}')
        
        elif plot_type == 'violin_plot':
            sns.violinplot(data=df, x=target_col, y=col, hue=target_col, legend=False, palette='Set2')
            plt.title(f'Violin plot of {col} by {target_col}')
            
    plt.tight_layout()
    plt.show()
    
def plot_bivariate_cat_target(df, cat_cols, target_col='label'):
    total_plot = len(cat_cols)
    cols = 3
    rows = math.ceil(total_plot / cols)
    
    plt.figure(figsize=(5 * cols, 5 * rows))
    for i, col in enumerate(cat_cols):
        plt.subplot(rows, cols, i + 1)
        sns.countplot(data=df, x=col, hue=target_col, palette='Set2')
        plt.title(f'Count plot of {col} by {target_col}')
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    plt.show()
    
def perform_chi_square_test(df, cat_cols, target_col='label'):
    print(f"--- TABLE OF CHI-SQUARE TESTING (Target: {target_col}) ---\n")
    print(f"{'Feature':<25} | {'p-value':<20} | {'Conclusion'}")
    print("-" * 75)
    
    results = []
    
    for col in cat_cols:
        if col == target_col:
            continue
            
        try:
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            if p < 0.05:
                is_significant = "Has an effect"
            else:
                is_significant = "No effect"
            
            results.append({
                'Feature': col,
                'p-value': p,
                'Conclusion': is_significant
            })
        except Exception as e:
            pass 
            
    results_df = pd.DataFrame(results).sort_values(by='p-value').reset_index(drop=True)
    
    for index, row in results_df.iterrows():
        p_str = f"{row['p-value']:.5e}" if row['p-value'] < 0.001 else f"{row['p-value']:.4f}"
        print(f"{row['Feature']:<25} | {p_str:<20} | {row['Conclusion']}")
        
    return results_df


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_delay_rate(df, cat_cols, target_col='label'):
    # Only keep categorical columns with less than 20 unique values to avoid crashing/clutter
    valid_cols = [col for col in cat_cols if df[col].nunique() < 20 and col != target_col]
    
    total_plots = len(valid_cols)
    cols = 2
    rows = math.ceil(total_plots / cols)
    
    plt.figure(figsize=(16, 6 * rows))
    
    # Calculate the overall average delay rate of the dataset (The Baseline)
    overall_delay_rate = df[target_col].mean() * 100 
    
    for i, col in enumerate(valid_cols):
        plt.subplot(rows, cols, i + 1)
        
        # Calculate delay rate (Label = 1) for each group in the column
        delay_rates = df.groupby(col)[target_col].mean() * 100
        
        # Plot the bar chart for delay rates
        ax = sns.barplot(x=delay_rates.index, y=delay_rates.values, color='#F08C6A', edgecolor='black')
        
        # Draw the red baseline (Overall company average)
        plt.axhline(overall_delay_rate, color='red', linestyle='--', linewidth=2, 
                    label=f'Company Avg: {overall_delay_rate:.2f}%')
        
        # Formatting and Decoration
        plt.title(f'Delay Rate by {col}', fontsize=14, fontweight='bold')
        plt.ylabel('Delay Rate (%)', fontsize=12)
        plt.xlabel(col, fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        
        # Add percentage numbers on top of each bar for readability
        for p in ax.patches:
            if p.get_height() > 0: # Only annotate if the rate is > 0
                ax.annotate(f'{p.get_height():.2f}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 8), 
                            textcoords = 'offset points',
                            fontsize=10)
                        
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, num_cols, method='spearman', threshold=0.8):
    """
    Plots a correlation heatmap and automatically detects multicollinearity.
    """
    print(f"Calculating correlation matrix using '{method.upper()}' method")
    
    # Calculate the correlation matrix
    corr_matrix = df[num_cols].corr(method=method)
    
    # Create a mask to hide the upper triangle (to avoid duplicate visual information)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot the heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, 
                mask=mask, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                vmin=-1, vmax=1, 
                square=True, 
                linewidths=.5,
                cbar_kws={"shrink": .8})
    
    plt.title(f"Correlation Heatmap ({method.capitalize()})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"MULTICOLLINEARITY WARNING SYSTEM (Threshold {threshold}) ")

    high_corr_found = False
    # Iterate through the lower triangle of the matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_corr_found = True
                colname_1 = corr_matrix.columns[i]
                colname_2 = corr_matrix.columns[j]
                print(f"Feature Pair: [{colname_1}] and [{colname_2}]")
                print(f"Correlation level: {corr_value:.2f}")
                print(f"Recommendation: Consider DROPPING one of these two features\n")
                
    if not high_corr_found:
        print("No severe multicollinearity detected among the features.")
