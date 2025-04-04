# import necessary libraries for feature selection
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from happy_customers.data.load_data import load_customer_survey

def feature_selection_customer(customer_survey):
    # Split features from target variable
    features = customer_survey.drop(columns=['Target'])
    target_variable = customer_survey['Target']
    np.random.seed(42)
    # Standardize the features for Lasso
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)

    # Compute Mutual Information (MI) between features and target variable
    mi_scores = mutual_info_classif(features, target_variable, random_state=42)
    mi_df = pd.DataFrame({'Feature':features.columns, 'Mutual Information': mi_scores}).sort_values(
        by='Mutual Information', ascending=False
    )
    # Compute ANOVA F-test scores
    anova_f_values, _ = f_classif(features, target_variable)
    anova_df = pd.DataFrame({'Feature': features.columns, 'ANOVA F-Score': anova_f_values}).sort_values(
        by='ANOVA F-Score', ascending=False
    )
    # Fit Lasso Regularized Logistic Regression (L1 Regularization)
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.2)
    # Train logistic regression model
    lasso.fit(features_standardized, target_variable)
    # Extract and sort the model's coefficients
    lasso_df = pd.DataFrame({
        'Feature':features.columns, 
        'Lasso Coefficients':lasso.coef_[0]
        }).sort_values(by='Lasso Coefficients', key=abs, ascending=False)

    # Fit Random Forest 
    random_forest = RandomForestClassifier(n_estimators=100, random_state=50)
    # Train model
    random_forest.fit(features, target_variable)
    # Extract and sort rf model's coefficients
    random_forest_df = pd.DataFrame({
        'Feature':features.columns,
        'Random Forest Importance':random_forest.feature_importances_
    }).sort_values(by='Random Forest Importance', ascending=False)

    # Combine all feature selection dataframes
    combined_feature_selection = mi_df.merge(anova_df, on='Feature').merge(lasso_df,on='Feature').merge(
        random_forest_df, on='Feature'
    )
    print(combined_feature_selection)

# Call the feature selection
if __name__ == "__main__":
    customer_survey = load_customer_survey()
    feature_selection_customer(customer_survey)