import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


import pandas as pd

def engineer_date_features(df, date_cols=['Order date', 'VSD']):
    """
    Extract useful information from date columns and remove the original columns.
    """
    df_out = df.copy()

    for col in date_cols:
        df_out[col] = pd.to_datetime(df_out[col], errors='coerce')
        
    # Distance between the order date and the expected delivery date
    if 'Order date' in df_out.columns and 'VSD' in df_out.columns:
        df_out['Expected_Lead_Time'] = (df_out['VSD'] - df_out['Order date']).dt.days

    # Extract Month (to capture year-end trends)
    df_out['Order_Month'] = df_out['Order date'].dt.month
    
    # Extract Day of Week (0 = Monday, 6 = Sunday)
    df_out['Order_DayOfWeek'] = df_out['Order date'].dt.dayofweek
    
    # Is it a weekend order? (0 = No, 1 = Yes)
    df_out['Is_Weekend_Order'] = df_out['Order_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Extract similar features for VSD 
    df_out['VSD_Month'] = df_out['VSD'].dt.month
    
    # REMOVE ORIGINAL COLUMNS: Drop object/datetime types, keep the new numerical features
    df_out = df_out.drop(columns=date_cols)
    
    # Calculate the number of new features created for the log message
    new_features_count = len(df_out.columns) - (len(df.columns) - len(date_cols))
    print(f"Datetime processing complete: Created {new_features_count} new feature columns.")
    
    return df_out

def apply_log_transform(df, cols_to_transform):
    df_out = df.copy()
    
    for col in cols_to_transform:
        if (df_out[col] < 0).any():
            print(f"Column '{col}' contains negative values. Skipping log transformation for this column.")
            continue

        df_out[col] = np.log1p(df_out[col])
        
    print(f"Successfully applied Log Transformation to {len(cols_to_transform)} columns.")
    
    return df_out


def split_into_train_dev_test(df, target_col, test_size=0.1, dev_size=0.1, random_state=42, stratify=True):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(test_size + dev_size), 
        random_state=random_state,
        stratify=y if stratify else None
    )
    
    test_proportion = test_size / (test_size + dev_size)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=test_proportion, 
        random_state=random_state,
        stratify=y_temp if stratify else None
    )
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def split_into_train_test(df, target_col, test_size=0.2, random_state=42, stratify=True):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y if stratify else None
    )
    return X_train, X_test, y_train, y_test

def build_custom_processor(target_encode, one_hot_encode, num_cols):
    # Handling numerical columns
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Handling categorical columns
    target_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target', TargetEncoder())
    ])

    onehot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('target', target_transformer, target_encode),
            ('onehot', onehot_transformer, one_hot_encode)
        ],
        remainder='passthrough' 
    )

    return preprocessor



def train_and_evaluate(model, X_train, y_train, X_dev, y_dev, X_test, y_test, preprocessor):
    """
    Wraps the entire workflow: Preprocessing -> Data Balancing -> Training -> Comprehensive Evaluation.
    Includes a DEV set to monitor Overfitting.
    """
    # 1. Configure data balancing ratios
    # Undersample the majority class so the minority is 50% of the majority
    undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    # Oversample the minority class so it reaches 30% of the majority
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    
    # 2. Assemble the Pipeline
    my_pipeline = ImbPipeline(steps=[
        ('preprocess', preprocessor),
        ('smote', smote),
        ('undersample', undersample),
        ('classifier', model)
    ])
    
    # 3. Training Process 
    print(f"Training model: {model.__class__.__name__}...")
    my_pipeline.fit(X_train, y_train)
    
    # 4. EVALUATION ON DEV SET (VALIDATION)
    print("\n" + "="*50)
    print("DEV SET REPORT (VALIDATION) ")
    print(" Used for monitoring Overfitting and Hyperparameter Tuning")
    print("="*50)
    y_dev_pred = my_pipeline.predict(X_dev)
    print(classification_report(y_dev, y_dev_pred, digits=4))
    
    if hasattr(my_pipeline, "predict_proba"):
        auc_dev = roc_auc_score(y_dev, my_pipeline.predict_proba(X_dev)[:, 1])
        print(f"[+] AUC-ROC (Dev): {auc_dev:.4f}")

    # 5. EVALUATION ON TEST SET 
    print("\n" + "="*50)
    print(" TEST SET REPORT (FINAL RESULT) ")
    print(" Final performance metrics for project reporting")
    print("="*50)
    y_test_pred = my_pipeline.predict(X_test)
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred, digits=4))
    
    if hasattr(my_pipeline, "predict_proba"):
        auc_test = roc_auc_score(y_test, my_pipeline.predict_proba(X_test)[:, 1])
        print(f"AUC-ROC (Test): {auc_test:.4f}")
        
    return my_pipeline

def train_A_test_B(model, X_train, y_train, X_test, y_test, preprocessor):
    # 1. Configure data balancing ratios
    undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    
    # 2. Assemble the Pipeline
    my_pipeline = ImbPipeline(steps=[
        ('preprocess', preprocessor),
        ('smote', smote),
        ('undersample', undersample),
        ('classifier', model)
    ])
    
    # 3. Training Process 
    print(f"Training model: {model.__class__.__name__}...")
    my_pipeline.fit(X_train, y_train)
    
    # 4. Evaluation on test set
    print("\n" + "="*50)
    print(" TEST SET REPORT (FINAL RESULT) ")
    print(" Final performance metrics for project reporting")
    print("="*50)
    y_test_pred = my_pipeline.predict(X_test)
    print("Confusion Matrix (Test)")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred, digits=4))
    
    if hasattr(my_pipeline, "predict_proba"):
        auc_test = roc_auc_score(y_test, my_pipeline.predict_proba(X_test)[:, 1])
        print(f"AUC-ROC (Test): {auc_test:.4f}")
        
    return my_pipeline

from sklearn.model_selection import StratifiedKFold

def k_fold(model, X_A, y_A, X_B, y_B, preprocessor, k=5, random_state=42):
    # A + B
    X_all = pd.concat([X_A, X_B], axis=0).reset_index(drop=True)
    y_all = pd.concat([y_A, y_B], axis=0).reset_index(drop=True)

    print(f"Concat Dataset: {X_all.shape[0]} samples | "
          f"Class ratio: {y_all.value_counts(normalize=True).to_dict()}")
    print("=" * 50)

    # Stratified K-Fold 
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    fold_aucs, fold_f1s = [], []
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
        print(f"\n{'='*20} FOLD {fold_idx}/{k} {'='*20}")

        X_train_fold, X_test_fold = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train_fold, y_test_fold = y_all.iloc[train_idx], y_all.iloc[test_idx]

        # pipeline (SMOTE → Undersample → model)
        undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)
        smote       = SMOTE(sampling_strategy=0.3, random_state=random_state)

        pipeline = ImbPipeline(steps=[
            ('preprocess',   preprocessor),
            ('smote',        smote),
            ('undersample',  undersample),
            ('classifier',   model)
        ])

        # Train
        print(f"Training model: {model.__class__.__name__}...")
        pipeline.fit(X_train_fold, y_train_fold)

        # Evaluate on fold test
        y_pred = pipeline.predict(X_test_fold)

        report = classification_report(y_test_fold, y_pred, digits=4, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]
        fold_f1s.append(macro_f1)

        print(f"Confusion Matrix (Fold {fold_idx}):")
        print(confusion_matrix(y_test_fold, y_pred))
        print(f"\nClassification Report (Fold {fold_idx}):")
        print(classification_report(y_test_fold, y_pred, digits=4))

        auc = None
        if hasattr(pipeline, "predict_proba"):
            auc = roc_auc_score(y_test_fold, pipeline.predict_proba(X_test_fold)[:, 1])
            fold_aucs.append(auc)
            print(f"AUC-ROC (Fold {fold_idx}): {auc:.4f}")

        fold_results.append({
            "fold":     fold_idx,
            "macro_f1": macro_f1,
            "auc":      auc
        })

    # Conclusion
    print("\n" + "=" * 50)
    print(f"Conclusion {k}-FOLD CROSS VALIDATION")
    print("=" * 50)
    print(f"Macro-F1  | Per fold: {[f'{v:.4f}' for v in fold_f1s]}")
    print(f"           Avg: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    if fold_aucs:
        print(f"AUC-ROC   | Per fold: {[f'{v:.4f}' for v in fold_aucs]}")
        print(f"           Avg: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    return {
        "fold_results": fold_results,
        "avg_macro_f1": np.mean(fold_f1s),
        "std_macro_f1": np.std(fold_f1s),
        "avg_auc":      np.mean(fold_aucs) if fold_aucs else None,
        "std_auc":      np.std(fold_aucs)  if fold_aucs else None,
    }
    
def exp4_train_AkB_test_remaining_B(
    model, X_A, y_A, X_B, y_B, preprocessor,
    k_percents=[0.1, 0.3, 0.5, 0.7, 0.9],
    random_state=42
):

    all_results = []

    for k in k_percents:
        print(f"\n{'='*55}")
        print(f"  K = {int(k*100)}% | Train: A + {int(k*100)}%B  →  Test: {int((1-k)*100)}%B")
        print(f"{'='*55}")

        # 1. Split B into k% (train) and (1-k)% (test)
        X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(
            X_B, y_B,
            train_size=k,
            random_state=random_state,
            stratify=y_B          # Maintain class ratio
        )

        # 2. Combine A + k%B as the training set
        X_train = pd.concat([X_A, X_B_train], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_A, y_B_train], axis=0).reset_index(drop=True)

        print(f"  Train size : {len(X_train):,} samples  "
              f"(A={len(X_A):,} + {int(k*100)}%B={len(X_B_train):,})")
        print(f"  Test size  : {len(X_B_test):,} samples ({int((1-k)*100)}%B)")

        # 3. Build pipeline
        smote       = SMOTE(sampling_strategy=0.3, random_state=random_state)
        undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)

        pipeline = ImbPipeline(steps=[
            ('preprocess',  preprocessor),
            ('smote',       smote),
            ('undersample', undersample),
            ('classifier',  model)
        ])

        # 4. Train
        print(f"\n  Training {model.__class__.__name__}...")
        pipeline.fit(X_train, y_train)

        # 5. Evaluate on (1-k)%B
        y_pred = pipeline.predict(X_B_test)

        report   = classification_report(y_B_test, y_pred, digits=4, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]

        print(f"\n  Confusion Matrix:")
        print(confusion_matrix(y_B_test, y_pred))
        print(f"\n  Classification Report:")
        print(classification_report(y_B_test, y_pred, digits=4))

        auc = None
        if hasattr(pipeline, "predict_proba"):
            auc = roc_auc_score(y_B_test, pipeline.predict_proba(X_B_test)[:, 1])
            print(f"  AUC-ROC: {auc:.4f}")

        all_results.append({
            "k_percent"     : int(k * 100),
            "train_size"    : len(X_train),
            "test_size"     : len(X_B_test),
            "macro_f1"      : macro_f1,
            "auc"           : auc,
        })

    # 6. Summary Table
    print(f"\n{'='*55}")
    print(f"  EXP4 SUMMARY — {model.__class__.__name__}")
    print(f"{'='*55}")
    print(f"  {'K%':>5} | {'Train size':>12} | {'Test size':>10} | {'Macro-F1':>10} | {'AUC':>8}")
    print(f"  {'-'*55}")
    for r in all_results:
        auc_str = f"{r['auc']:.4f}" if r['auc'] else "N/A"
        print(f"  {r['k_percent']:>4}% | {r['train_size']:>12,} | {r['test_size']:>10,} | "
              f"{r['macro_f1']:>10.4f} | {auc_str:>8}")

    return all_results