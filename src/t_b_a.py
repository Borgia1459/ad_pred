import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier as xgb
from sklearn.linear_model import SGDClassifier
import config  # Import the config module
import os

class AdClick:
    def __init__(self):
        try:
            # Construct the absolute path to the training data file
            training_path = os.path.join(os.path.dirname(__file__), config.TRAINING)
            self.data = pd.read_csv(training_path)
            print(self.data.head())
            self.target = 'Clicked on Ad'
            
            # Assuming 'Clicked' is the target column
            self.x = self.data.drop(self.target, axis=1)
            self.y = self.data[self.target]
            
            print(f'Y shape: {self.y.shape}')
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.1, random_state=42)
            
            # Apply the OneHotEncoder on the training data
            self.enc = OneHotEncoder(handle_unknown='ignore')
            self.x_train_enc = self.enc.fit_transform(self.x_train)
            
            # Apply the transform function on the test data
            self.x_test_enc = self.enc.transform(self.x_test)
            
            # Define the parameter grid for GridSearchCV
            self.param = {'max_depth': [3, 10, None]}
            
        except FileNotFoundError:
            print(f"File not found: {training_path}")
        except Exception as e:
            print(f'An error occurred while loading the data: {e}')

    def train_decision_tree(self):
        try:
            # Use the gini coefficient for decision tree model
            self.decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30, ccp_alpha=1, random_state=42)
            self.gridSearch = GridSearchCV(self.decision_tree, self.param, n_jobs=-1, cv=3, scoring='roc_auc')
            self.gridSearch.fit(self.x_train_enc, self.y_train)
            
            print(f'Grid search results for Decision Tree: {self.gridSearch.best_params_}')
            self.decision_tree_best = self.gridSearch.best_estimator_
            
            self.pos_prob = self.decision_tree_best.predict_proba(self.x_test_enc)[:, 1]
            print(f'The ROC AUC on the test data is: {roc_auc_score(self.y_test, self.pos_prob)}')
        except Exception as e:
            print(f'An error occurred during Decision Tree training: {e}')

    def sgd_logistic_regression(self):
        try: 
            #initialize the SGDClassiffier with logistic reg
            self.sgd_log = SGDClassifier(penalty='l1', alpha=0.001,fit_intercept= True,learning_rate='constant',loss='log_loss',max_iter= 1, tol=1e-3, random_state=42)
            
            #fit the model to the training data
            self.sgd_log.fit(self.x_train_enc, self.y_train) 
            
            #predict probabilities
            self.pos_prob = self.sgd_log.predict_proba(self.x_test_enc)[:, 1] 
            
            #Calculate ROC AUC score
            print(f'The ROC AUC on the test data is: {roc_auc_score(self.y_test,self.pos_prob):.3f}')
        except Exception as e:
            print(f'An error occurred during SGD logistic regression: {e}')

    def train_random_forest(self):
        try:
            # Use the random forest model
            self.random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=15, ccp_alpha=0.001)
            self.grid_search_rf = GridSearchCV(self.random_forest, self.param, n_jobs=-1, cv=3, scoring='roc_auc')
            self.grid_search_rf.fit(self.x_train_enc, self.y_train)
            
            print(f'Grid search results for Random Forest: {self.grid_search_rf.best_params_}')
            self.random_forest_best = self.grid_search_rf.best_estimator_
            
            self.pos_prob_rf = self.random_forest_best.predict_proba(self.x_test_enc)[:, 1]
            print(f'The ROC AUC on the test data for Random Forest is: {roc_auc_score(self.y_test, self.pos_prob_rf)}')
        except Exception as e:
            print(f'An error occurred during Random Forest training: {e}')

    def train_xgboost(self):
        try:
            # Ensembling with decision trees using XGBoost
            self.model = xgb(learning_rate=0.1, n_estimators=100, max_depth=10)
            self.model.fit(self.x_train_enc, self.y_train)
            self.pos_prob = self.model.predict_proba(self.x_test_enc)[:, 1]
            print(f'The ROC AUC on the test data is, XGB: {roc_auc_score(self.y_test, self.pos_prob):.3f}')
        except Exception as e:
            print(f'An error occurred during XGBoost training: {e}')

    def logistic_regression(self):
        try:
            self.enc = OneHotEncoder(handle_unknown='ignore')
            self.x_train_enc = self.enc.fit_transform(self.x_train)
            self.x_test_enc = self.enc.transform(self.x_test)
            # Add logistic regression training code here
        except Exception as e:
            print(f'An error occurred during Logistic Regression training: {e}')

if __name__ == '__main__':
    adClickData = AdClick()
    #adClickData.train_decision_tree()
    #adClickData.train_random_forest()
    adClickData.train_xgboost()
    adClickData.sgd_logistic_regression()
    # adClickData.logistic_regression()  # Uncomment and implement logistic regression training if needed