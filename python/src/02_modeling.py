import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("imbalanced-learn not installed. partitioning without SMOTE.")
    ImbPipeline = Pipeline
    SMOTE = None

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from utils import ...

def load_data():
    return pd.read_parquet('data/processed/two_predictor_data.parquet')

def get_models():
    """
    Define models and their hyperparameter grids.
    Matching R: RF, SVM, NB, DT, Boosted Trees, KNN, Elasticnet, Ridge, Lasso.
    """
    models = {}
    
    # 1. Random Forest
    models['Random_Forest'] = {
        'model': RandomForestClassifier(random_state=1234),
        'params': {
            'classifier__n_estimators': [100, 200, 500],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__max_features': ['sqrt', 'log2']
        }
    }
    
    # 2. SVM
    models['SVM'] = {
        'model': SVC(probability=True, random_state=1234),
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear']
        }
    }
    
    # 3. Naive Bayes
    models['Naive_Bayes'] = {
        'model': GaussianNB(),
        'params': {} # GaussianNB has fewer params to tune usually
    }
    
    # 4. Decision Tree
    models['Decision_Tree'] = {
        'model': DecisionTreeClassifier(random_state=1234),
        'params': {
            'classifier__max_depth': [None, 5, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
    }
    
    # 5. Trimmed DT (Tree depth=2, min_n=2 in R)
    models['n2_Decision_Tree'] = {
        'model': DecisionTreeClassifier(max_depth=2, min_samples_split=2, random_state=1234),
        'params': {}
    }
    
    # 6. Boosted Trees (XGBoost)
    try:
        from xgboost import XGBClassifier
        models['Boosted_Trees'] = {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1234),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.3],
                'classifier__max_depth': [3, 6, 9]
            }
        }
    except ImportError:
        print("XGBoost not installed.")

    # 7. KNN
    models['KNN'] = {
        'model': KNeighborsClassifier(),
        'params': {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance']
        }
    }

    # 8. ElasticNet (LogisticRegression with penalty='elasticnet')
    models['Elasticnet'] = {
        'model': LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000, random_state=1234),
        'params': {
            'classifier__l1_ratio': [0.1, 0.5, 0.9],
            'classifier__C': [0.1, 1, 10]
        }
    }
    
    # 9. Ridge (l2)
    models['Ridge_Regression'] = {
        'model': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=1234),
        'params': {
            'classifier__C': [0.1, 1, 10]
        }
    }
    
    # 10. Lasso (l1)
    models['Lasso_Regression'] = {
        'model': LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=1234),
        'params': {
            'classifier__C': [0.1, 1, 10]
        }
    }
    
    return models

def get_preprocessors():
    """
    Define preprocessing steps (Scaling, Normalization, SMOTE, Corr).
    R recipes: basic, smote, scale, norm, corr, and their combinations.
    Here we define a list of (name, steps_list).
    """
    # steps: [('imputer', SimpleImputer(...)), ('scaler', ...), ('smote', ...)]
    # Note: 'corr' removal in sklearn is manual FeatureSelection. R's step_corr removes highly correlated features.
    # Since we have only 2 numerical predictors (row_m_z, buffer_percent), correlation might not be an issue or easily handled.
    # We will implement Basic, Scaled, Balanced (SMOTE), and Balanced+Scaled.
    
    preprocessors = {}
    
    # Basic: Just Impute?
    preprocessors['basic'] = [
        ('imputer', SimpleImputer(strategy='median'))
    ]
    
    # Scaled
    preprocessors['scaled'] = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
    
    # Balanced (SMOTE)
    if SMOTE:
        preprocessors['balanced'] = [
            ('imputer', SimpleImputer(strategy='median')),
            ('smote', SMOTE(random_state=1234))
        ]
        
        preprocessors['balanced_scaled'] = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=1234))
        ]
    
    return preprocessors

def run_modeling():
    df = load_data()
    
    # Predictors: row_m_z, buffer_percent
    # Outcome: is_lipids (Yes/No)
    X = df[['row_m_z', 'buffer_percent']]
    y = df['is_lipids'].map({'Yes': 1, 'No': 0}) # Encode outcome
    
    # Split
    # R: 3/4 training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1234)
    
    joblib.dump((X_train, X_test, y_train, y_test), 'data/processed/train_test_split.joblib')
    
    models = get_models()
    preprocessors = get_preprocessors()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    results = []
    best_models = {}
    
    for prep_name, prep_steps in preprocessors.items():
        for model_name, model_config in models.items():
            print(f"Training {model_name} with {prep_name}...")
            
            # Construct Pipeline
            # Note: ImbPipeline handles SMOTE during fit automatically
            steps = prep_steps + [('classifier', model_config['model'])]
            pipeline = ImbPipeline(steps)
            
            # Grid Search
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234) # 10 folds in R, 5 is faster for now
            clf = GridSearchCV(pipeline, model_config['params'], cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
            
            try:
                clf.fit(X_train, y_train)
                
                # Best score
                mean_acc = clf.best_score_
                
                # Test set eval
                y_pred = clf.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                test_roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
                
                result_entry = {
                    'preprocess': prep_name,
                    'model': model_name,
                    'best_cv_accuracy': mean_acc,
                    'test_accuracy': test_acc,
                    'test_roc_auc': test_roc,
                    'best_params': clf.best_params_
                }
                results.append(result_entry)
                
                # Save best estimator per model type (simplified)
                key = f"{prep_name}_{model_name}"
                joblib.dump(clf.best_estimator_, f"models/{key}.joblib")
                
            except Exception as e:
                print(f"Failed {model_name} + {prep_name}: {e}")

    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/modeling_results.csv', index=False)
    print(results_df.sort_values(by='test_accuracy', ascending=False).head())
    print("Modeling Done.")

if __name__ == "__main__":
    run_modeling()
