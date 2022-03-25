import pickle
import pandas as pd
import numpy as np
import streamlit as st

numericals_minmax = pickle.load(open('serialization/utilities/numericals_minmax.pickle', 'rb')) # NOTE: you can debug the min_max using st.json(min_max)
categoricals_map = pickle.load(open('serialization/utilities/categoricals_map.pickle', 'rb')) # NOTE: you can debug the min_max using st.json(min_max)
scaler_salary = pickle.load(open('serialization/utilities/scaler_salary.pickle', 'rb'))
scaler_balance = pickle.load(open('serialization/utilities/scaler_balance.pickle', 'rb'))
scaler_duration = pickle.load(open('serialization/utilities/scaler_duration.pickle', 'rb'))
scaler_pdays = pickle.load(open('serialization/utilities/scaler_pdays.pickle', 'rb'))

model_names = (
    'DecisionTreeClassifier',
    'LogisticRegression',
)

decision_tree_classifier_tuned = pickle.load(open('serialization/models/decision_tree_classifier_tuned.pickle', 'rb'))
logistic_regression_tuned = pickle.load(open('serialization/models/logistic_regression_tuned.pickle', 'rb'))

###### Serialized Dataframes
metrics_result = pickle.load(open('serialization/dataframes/metrics_result.pickle', 'rb'))

def predict(parameters, modelname):

    job = parameters['job']
    marital = parameters['marital']
    education = parameters['education']
    marital_education = marital + "-" + education
    previous = parameters['previous']
    salary = parameters['salary']
    balance = parameters['balance']
    duration = parameters['duration']
    pdays = parameters['pdays']

    inputs = pd.DataFrame({
        'salary': [scaler_salary.transform(np.array([salary]).reshape(1, -1))[0][0]],
        'balance': [scaler_balance.transform(np.array([balance]).reshape(1, -1))[0][0]],
        'duration': [scaler_duration.transform(np.array([duration]).reshape(1, -1))[0][0]],
        'pdays': [scaler_pdays.transform(np.array([pdays]).reshape(1, -1))[0][0]],

        'job': [categoricals_map['job'][job]],
        'marital': [categoricals_map['marital'][marital]],
        'education': [categoricals_map['education'][education]],
        'marital-education': [categoricals_map['marital-education'][marital_education]],
        'previous': [categoricals_map['previous'][previous]],
    })

    output = None
    response = None

    if (modelname == model_names[0]):

        output = decision_tree_classifier_tuned.predict(inputs)

    elif(modelname == model_names[1]):

        output = logistic_regression_tuned.predict(inputs)

    # if (output):
    response_decoder = {0: 'No', 1: 'Yes'}
    response = response_decoder[output[0]]

    return response