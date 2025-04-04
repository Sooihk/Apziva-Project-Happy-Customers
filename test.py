# import necessary libraries for modeling
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

# Load csv dataset
customer_survey_filepath = 'ACME-HappinessSurvey2020.csv'
customer_survey = pd.read_csv(customer_survey_filepath)

# Rename column names for better interpretability
customer_survey.rename(columns = {
    'Y':'Target',
    'X1':'Delivered_On_Time',
    'X2':'Contents_As_Expected',
    'X3':'Everything_Wanted_Ordered',
    'X4':'Good_Price',
    'X5':'Satisfied_With_Courier',
    'X6':'Ordering_Ease'
},inplace=True)
customer_survey.head()
customer_survey.to_csv('Apziva Project 1/data/interim/renamed_customer_survey.csv', index=False)

# ------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
# Create figure and axis with optimized size
fig, ax = plt.subplots(figsize=(6,4))

# Plot distribution of Target variable
bars = customer_survey['Target'].value_counts().plot(
    kind='bar',
    color=['steelblue','orange'],
    edgecolor='black',
    width=0.6,
    ax=ax
)

# Set labels and title with proper formatting
ax.set_xlabel('Target Attribute', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Distribution of Customer Happiness (1) and Unhappiness (0)', fontsize=16)
ax.yaxis.grid(True,linestyle='--',alpha=0.7)

# Add "Figure 1." label at the bottom of the figure
fig.text(0.5, -0.1, 'Figure 1.', fontsize=12, ha='center')

plt.show()

# ------------------------------------------------------------------------------------------------------
# Create figure and axes
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))  # Arrange subplots in a 2x3 grid
axes = axes.flatten()  # Flatten to easily iterate over

# Plot histograms for each feature
for i, column in enumerate(customer_survey.columns[1:]):  # Skip the first column (assuming it's the target)
    ax = axes[i]  # Select subplot
    customer_survey[column].hist(ax=ax, bins=5, edgecolor='black', color='steelblue', alpha=0.8)

    # Formatting
    ax.set_title(f'{column}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)  # Add grid for better readability

# Adjust layout and add a main title
fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust spacing to fit the title
# Add "Figure 1." label at the bottom of the figure
fig.text(0.5, -0.02, 'Figure 2.', fontsize=12, ha='center')
# Show plot
plt.show()
# ------------------------------------------------------------------------------------------------------
import seaborn as sns

correlation_matrix = customer_survey.corr()

# Plotting correlation heatmap
fig, ax = plt.subplots(figsize=(10,7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidth=0.5,
    square=True, # heatmap cells are square
    cbar_kws={'shrink':0.8},
    ax=ax
)
# Set Title
ax.set_title('Feature Correlation Heatmap', fontsize=16)
# Tick label formatting, rotate x-axis label for better visibility
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12, rotation=0)

# Adjust layout to prevent clipping
fig.tight_layout()
# Show the plot
plt.show()

# ------------------------------------------------------------------------------------------------------

# import necessary libraries for feature selection
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

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
combined_feature_selection


# ------------------------------------------------------------------------------------------------------

# Selecting features and target
selected_features = ['Delivered_On_Time', 'Satisfied_With_Courier', 'Everything_Wanted_Ordered', 'Ordering_Ease']
x = customer_survey[selected_features]
y = customer_survey['Target']
random_seed = 3

# Splitting Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.14, random_state=random_seed)
x_train.to_csv('Apziva Project 1/data/processed/final_x_train.csv', index=False)
y_train.to_csv('Apziva Project 1/data/processed/final_y_train.csv', index=False)
x_test.to_csv('Apziva Project 1/data/processed/final_x_test.csv', index=False)
y_test.to_csv('Apziva Project 1/data/processed/final_y_test.csv', index=False)


# Define models to be used
models = {
    'Logistic Regression': LogisticRegression(random_state=random_seed),
    'Decision Tree': DecisionTreeClassifier(random_state=random_seed),
    'Random Forest': RandomForestClassifier(random_state=random_seed),
    'Support Vector Machine': SVC(),
    'Naive Bayes': GaussianNB()
}

# Initialize results containers
test_results = []
train_results = []

# for loop to train dataset on classification models
for name, model in models.items():
    model.fit(x_train, y_train)

    # Predictions on test and train data
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)

    # Test metrics
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    test_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': test_report['1']['precision'],
        'Recall': test_report['1']['recall'],
        'F1-Score': test_report['1']['f1-score']
    })

    # Training metrics
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    train_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_train, y_train_pred),
        'Precision': train_report['1']['precision'],
        'Recall': train_report['1']['recall'],
        'F1-Score': train_report['1']['f1-score']
    })

# Create DataFrames
test_metrics = pd.DataFrame(test_results).sort_values(by='F1-Score', ascending=False)
train_metrics = pd.DataFrame(train_results).sort_values(by='F1-Score', ascending=False)

# Output both DataFrames
print("Test Set Performance")
display(test_metrics)

print("\n Training Set Performance")
display(train_metrics)

# ------------------------------------------------------------------------------------------------------

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
display(tunned_test_metrics)

# ------------------------------------------------------------------------------------------------------

# Merge on 'Model' column suffixes differentiate the untuned and tuned F1 scores
combined_f1_scores = test_metrics[['Model', 'F1-Score']].merge(
    tunned_test_metrics[['Model', 'F1-Score']],
    on='Model',
    suffixes=('_Untuned', '_Tuned')
)

# Sort by f1-score
combined_f1_scores = combined_f1_scores.sort_values(by='F1-Score_Tuned', ascending=False)

combined_f1_scores