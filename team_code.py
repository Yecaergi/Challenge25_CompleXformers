#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from helper_code import *
from entropiaWavelet import entropiaWavelet

def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = np.zeros((num_records, 8), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record)
        labels[i] = load_label(record)

    # Train the models.
    if verbose:
        print('Training the model on the data...')
    
    

    # Define the parameters for the random forest classifier and regressor.
    n_estimators = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state = 56  # Random state; set for reproducibility.

    # Fit the model.
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

def load_model(model_folder, verbose=True):
    model_filename = os.path.join(model_folder, 'model.sav')
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f'Modelo no encontrado en {model_filename}')
    return joblib.load(model_filename)

def run_model(record, model, verbose):
    # Load the model.
    model = model['model']

    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

def extract_features(record):
    try:
        signal, fields = load_signals(record)
        qrs = signal[:, 0]
        
        wavelet_features = entropiaWavelet(qrs)
        if wavelet_features is None:
            raise ValueError("La función entropiaWavelet devolvió None")
        
        mHd = wavelet_features.get('mHd', np.nan)
        mCd = wavelet_features.get('mCd', np.nan)
        
        header = load_header(record)
        age = get_age(header)
        sex = get_sex(header)
        
        one_hot_sex = np.array([sex == 'Female', sex == 'Male', sex not in ['Female', 'Male']], dtype=int)
        signal_mean = np.nanmean(signal) if np.isfinite(signal).sum() > 0 else 0.0
        signal_std = np.nanstd(signal) if np.isfinite(signal).sum() > 1 else 0.0
        
        features = np.concatenate(([age], one_hot_sex, [signal_mean, signal_std, mHd, mCd]))
        return features.astype(np.float32)
    except Exception as e:
        print(f'Error extrayendo características de {record}: {e}')
        return None


def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    
    print(f"Intentando guardar el modelo en: {filename}")  
    joblib.dump(d, filename, protocol=0)
    print(f"Modelo guardado exitosamente en: {filename}")  
