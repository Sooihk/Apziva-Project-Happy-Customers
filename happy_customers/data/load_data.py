import pandas as pd
from pathlib import Path

def load_customer_survey():
    customer_survey_filepath = Path.cwd() / 'data' / 'raw' / 'ACME-HappinessSurvey2020.csv'
    # Load csv dataset
    #customer_survey_filepath = 'Apziva Project 1/data/raw/ACME-HappinessSurvey2020.csv'
    #customer_survey_filepath = 'ACME-HappinessSurvey2020.csv'
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

    return customer_survey