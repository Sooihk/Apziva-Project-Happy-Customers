from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from happy_customers.data.load_data import load_customer_survey

def customer_happiness_model_predictions(customer_survey):

    # Selecting features and target
    selected_features = ['Delivered_On_Time', 'Satisfied_With_Courier', 'Everything_Wanted_Ordered', 'Ordering_Ease']
    x = customer_survey[selected_features]
    y = customer_survey['Target']
    random_seed = 3

    # Splitting Dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.14, random_state=random_seed)
    x_train.to_csv('data/processed/final_x_train.csv', index=False)
    y_train.to_csv('data/processed/final_y_train.csv', index=False)
    x_test.to_csv('data/processed/final_x_test.csv', index=False)
    y_test.to_csv('data/processed/final_y_test.csv', index=False)

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
    print(test_metrics)

    print("\nTraining Set Performance")
    print(train_metrics)

# Call the model baseline
if __name__ == "__main__":
    customer_survey = load_customer_survey()
    customer_happiness_model_predictions(customer_survey)