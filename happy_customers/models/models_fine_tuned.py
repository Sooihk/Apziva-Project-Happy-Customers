from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from happy_customers.data.load_data import load_customer_survey

def customer_happiness_finetuned_model_predictions(customer_survey):
    # Feature selection
    selected_features = ['Delivered_On_Time', 'Satisfied_With_Courier', 'Everything_Wanted_Ordered', 'Ordering_Ease']
    x = customer_survey[selected_features]
    y = customer_survey['Target']
    random_seed = 3

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.14, random_state=random_seed)

    # Define hyperparameter grids for each model, use clf__parameter format due to using Pipeline
    param_grids = {
        'Logistic Regression': {
            'clf__C': [0.01, 0.1, 1, 10], # inverse of regularizaton strength, lower values = more regularization
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear']  # supports l2
        },

        'Decision Tree': {
            'clf__max_depth': [3, 5, 10, None], # How deep the tree can go
            'clf__min_samples_split': [2, 5, 10], # Minimum number of samples needed to split an internal node
            'clf__min_samples_leaf': [1, 2, 4] # Minimum number of samples in a leaf node
        },

        'Random Forest': {
            'clf__n_estimators': [50, 100, 150], # Number of trees in forest, more trees leads to better generalization
            'clf__max_depth': [3, 5, 10, None], # Tree depth limit per tree
            'clf__min_samples_split': [2, 5], # threshold for when a node should split
            'clf__min_samples_leaf': [1, 2], # mininum number of samples required in a leaf
            'clf__max_features': ['sqrt', 'log2'], # number of features to consider at each split
            'clf__bootstrap' : [True, False]
        },

        'Support Vector Machine': {
            'clf__C': [0.1, 1, 10], # regularization parameter, lower values = more generalization
            'clf__kernel': ['linear', 'rbf'], # linear = linear boundary, rbf = non-linear boundary
            'clf__gamma': ['scale', 'auto']
        }
    }

    # Initialize models with pipelines for scaling when needed, using model Pipeline as it compelments GridSearchCV and cross validation.
    model_defs = {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=random_seed))]),
        'Decision Tree': Pipeline([('clf', DecisionTreeClassifier(random_state=random_seed))]),
        'Random Forest': Pipeline([('clf', RandomForestClassifier(random_state=random_seed))]),
        'Support Vector Machine': Pipeline([('scaler', StandardScaler()), ('clf', SVC())]),
        'Naive Bayes': Pipeline([('clf', GaussianNB())])  # No hyperparameter tuning needed
    }

    # Initialize results containers
    test_results = []

    # Loop through model presets
    for name, pipeline in model_defs.items():
        print(f"Tuning: {name}")

        # for models with parameter grids, perform exhaustive hyperparameter search using 5-fold cross validation.
        if name in param_grids:
            # Use GridSearchCV for models with hyperparameters, optmize for F1-Score during search
            grid = GridSearchCV(pipeline, param_grids[name], scoring='f1', cv=5, n_jobs=-1)
            grid.fit(x_train, y_train) 
            # obtain model trained with best hyperparameters during grid search
            best_model = grid.best_estimator_
        else:
            # No tuning for Naive Bayes
            pipeline.fit(x_train, y_train)
            best_model = pipeline

        # Predict on test subdataset
        y_test_pred = best_model.predict(x_test)

        # Test metrics
        test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        test_results.append({
            'Model': name,
            "Best Params": grid.best_params_,
            'Accuracy': accuracy_score(y_test, y_test_pred),
            'Precision': test_report['1']['precision'],
            'Recall': test_report['1']['recall'],
            'F1-Score': test_report['1']['f1-score']
        })


    # Create results DataFrames
    tunned_test_metrics = pd.DataFrame(test_results).sort_values(by='F1-Score', ascending=False)

    # Output both DataFrames
    print("\n Test Set Performance")
    print(tunned_test_metrics)

    # Call the model baseline
if __name__ == "__main__":
    customer_survey = load_customer_survey()
    customer_happiness_finetuned_model_predictions(customer_survey)