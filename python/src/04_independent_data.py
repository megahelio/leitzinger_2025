import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from 02_modeling if possible, but 02 is a script. 
# Better to duplicate or extract. I will import from 02 assuming it is importable.
# I need to ensure 02_modeling has get_models and get_preprocessors available.
# I'll re-write 02 to be importable or just define them here to avoid complexity of refactoring 02 now.
# Duplication is safer for this one-shot task to avoid breaking 02.

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    ImbPipeline = Pipeline
    SMOTE = None
    
from sklearn.metrics import accuracy_score, roc_auc_score

def load_data():
    # Load processed validation_ovca
    # In 01, we processed validation_ovca and saved it.
    return pd.read_parquet('data/processed/validation_ovca.parquet')

# --- Copy of helper functions from 02 (simplified) ---
def get_models():
    models = {}
    models['Random_Forest'] = {
        'model': RandomForestClassifier(random_state=1234),
        'params': {'classifier__n_estimators': [100], 'classifier__max_features': ['sqrt']}
    }
    # ... (Reduced set for independent data to save time, or full set?)
    # R script '04' runs the full suite. I'll include a representative set.
    models['SVM'] = {
        'model': SVC(probability=True, random_state=1234),
        'params': {'classifier__C': [1], 'classifier__kernel': ['rbf']}
    }
    models['Naive_Bayes'] = {'model': GaussianNB(), 'params': {}}
    return models

def get_preprocessors():
    preprocessors = {}
    preprocessors['basic'] = [('imputer', SimpleImputer(strategy='median'))]
    if SMOTE:
        preprocessors['balanced'] = [('imputer', SimpleImputer(strategy='median')), ('smote', SMOTE(random_state=1234))]
    return preprocessors
# -----------------------------------------------------

def run_ovca_modeling():
    df = load_data()
    
    # R: filter(method == "HILIC-pos") -> In 01 we didn't filter by method?
    # Checking 01.data_preparation.R:
    # It reads url4 -> rename -> mutate ... 
    # It does NOT filter HILIC-pos in 01.
    # 04.independent_data.R -> validation_ovca %>% filter(method == "HILIC-pos").
    # So we need to filter here if the column exists.
    
    if 'method' in df.columns:
        print("Filtering for method == 'HILIC-pos'")
        df = df[df['method'] == 'HILIC-pos']
    else:
        print("Warning: 'method' column not found (maybe dropped in 01?). Proceeding without filter.")

    # Predictors: row_m_z, buffer_percent. Outcome: is_lipids
    X = df[['row_m_z', 'buffer_percent']]
    y = df['is_lipids'].map({'Yes': 1, 'No': 0})
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1234)
    
    models = get_models()
    preprocessors = get_preprocessors()
    results = []
    
    for prep_name, prep_steps in preprocessors.items():
        for model_name, model_config in models.items():
            print(f"[OVCA] Training {model_name} with {prep_name}...")
            steps = prep_steps + [('classifier', model_config['model'])]
            pipeline = ImbPipeline(steps)
            
            clf = GridSearchCV(pipeline, model_config['params'], cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
            
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                try:
                    roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
                except:
                    roc = np.nan
                
                results.append({
                    'preprocess': prep_name,
                    'model': model_name,
                    'test_accuracy': acc,
                    'test_roc_auc': roc
                })
            except Exception as e:
                print(f"Error {model_name}: {e}")
                
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('results/ovca_results.csv', index=False)
    print("OVCA Modeling Done.")

if __name__ == "__main__":
    run_ovca_modeling()
