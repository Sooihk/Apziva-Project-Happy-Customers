# Happy Customers

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project aims to predict whether a customer is happy/unhappy based on their feedback in a logisitics and delivery context. Using survery responses covering delivery experience, courier satisfication, order fulfillment, and app usability, I apply classification models to uncover key drivers of customer satisfication. The project explores data through EDA, builds and tunes multiple machine learning models and evaluates them based on F1-score to guide operational improvements and enhance customer experience.

## Context:
We are one of the fastest growing startups in the logistics and delivery domain. We work
with several partners and make on-demand delivery to our customers. From operational
standpoint we have been facing several different challenges and everyday we are trying to
address these challenges.
We thrive on making our customers happy. As a growing startup, with a global expansion
strategy we know that we need to make our customers happy and the only way to do that is
to measure how happy each customer is. If we can predict what makes our customers happy
or unhappy, we can then take necessary actions.
Getting feedback from customers is not easy either, but we do our best to get constant
feedback from our customers. This is a crucial function to improve our operations across all
levels. We recently did a survey to a select customer cohort. You are presented with a subset
of this data. We will be using the remaining data as a private test set.

## Objective:
1. Predict if a customer is happy or not based on the answers they give to questions asked.
2. Reach 73% F1 score or above.
3. Identify features most important when predicting a customer's happiness.
4. Discover minimal set of features what would preserve the most information about the
problem while increasing predictabiliy of the data.

## Dataset:
Y = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers
X1 = my order was delivered on time
X2 = contents of my order was as I expected
X3 = I ordered everything I wanted to order
X4 = I paid a good price for my order
X5 = I am satisfied with my courier
X6 = the app makes ordering easy for me
Attributes X1 to X6 indicate the responses for each question and have values from 1 to 5
where the smaller number indicates less and the higher number indicates more towards the
answer.


## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         happy_customers and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── happy_customers   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes happy_customers a Python module
    │
    ├── data
        └── load_data.py        <- code to load the customer survey dataset
    │
    ├── features.py             <- Code for feature selection for customer survey dataset
    │
    ├── models                 
    │   ├── models_baseline.py          <- Code to run classification models to predict customer happiness based on features        
    │   └── models_fine_tuned.py        <- Code to run fine tuned GridSearch hyperparameters classification models to predict customer happiness based on features
    │
    └── plots.py                <- Code to create visualizations
```

--------
## Getting Started:  
Working with Python 3.12.2 for this project. 
Clone the repository and install the dependencies:

`pip install -r requirements.txt`

## Exploratory Data Analysis and Feature Selection:
To see the plots created in the EDA phase, run the following command:
  * A

`python -m happy_customers.plots`

Run the following command to see the statisical tests performed on the customer survey dataset to improve classification model performance:

`python -m happy_customers.features`

## Training and Evaluating Classification Models:
Results from Logisitic Regression, Decision Tree, Random Forest, Support Vector Machine, and Naive Bayes training on X1,X3,X5, and X6 features of the dataset using a .86/.14 training/test split with the following command:

`python -m happy_customers.models.models_baseline`

Results from Hyperparameter fine tuning previous models using GridSearchCV with the following command: 

`python -m happy_customers.models.models_fine_tuned`



