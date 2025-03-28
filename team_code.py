#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:10:37 2025

@author: gise
"""

from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os
import pywt  
from JSDiv import JSDiv
import matplotlib.pyplot as plt
from scipy.signal import resample, medfilt
import neurokit2 as nk
import scipy.signal as sig
import concurrent.futures
from scipy.signal import butter, freqz, filtfilt
from helper_code import *

# Parámetros fijos
dd, uu, ma_st = 30, 10, 2
sampling_new = 1000
k1, k2 = 48, 77
wNscales, wMaxScale = 16, 16
wScales = np.arange(1, wMaxScale + 1)

# Manejo simplificado de excepciones
class SignalProcessingException(Exception):
    pass

def obtener_picos(ecg_normalized, sampling_new):
    if len(ecg_normalized) < 50:
        return np.array([])
    try:
        _, rpeaks = nk.ecg_peaks(ecg_normalized, sampling_rate=sampling_new)
        return rpeaks['ECG_R_Peaks']
    except:
        return np.array([])

def filtro_mediana(ecg):
    kernel_size1 = min(201, len(ecg))
    kernel_size2 = min(601, len(ecg))
    estblwander = medfilt(ecg, kernel_size1)
    estblwander2 = medfilt(estblwander, kernel_size2)
    return ecg - estblwander2

def filtro_peine_DCyArmonicas(xx, DD=16, UU=2, MA_stages=2):
    hh_u = np.zeros(DD * UU)
    hh_u[::UU] = 1
    hh_u = np.flip(hh_u)
    xx_ci = xx[:(DD * UU)]
    yy_ci = np.sum(xx_ci * hh_u)
    yy = np.convolve(xx, hh_u, mode='same') / DD
    xx_aux = np.roll(xx, int((DD - 1) / 2 * MA_stages * UU))
    return xx_aux - yy

def filtrar_qrs(ventanas, median_window, threshold=0.98):
    return [w for w in ventanas if np.corrcoef(w, median_window)[0, 1] > threshold]

# Truco para incorporar las Db6 en pywt
ex = pywt.DiscreteContinuousWavelet('db6')
class DiscreteContinuousWaveletEx(type(ex)):
   def __init__(self, name=u'', filter_bank=None):
       super(type(ex), self)
       pywt.DiscreteContinuousWavelet.__init__(self, name, filter_bank)
       self.complex_cwt = False

def entropiaWavelet(signal_data, sampling):
    H = {'mHd': np.nan, 'mCd': np.nan}
    if signal_data is None or len(signal_data) < sampling or np.isnan(signal_data).any():
        return H

    ecg = resample(signal_data, int(len(signal_data) * (sampling_new / sampling)))

    try:
        ecg_filtered = filtro_peine_DCyArmonicas(ecg, DD=dd, UU=uu, MA_stages=ma_st)
        if np.all(ecg_filtered == 0):
            raise SignalProcessingException()
    except:
        ecg_filtered = filtro_mediana(ecg)

    ecg_cleaned = nk.ecg_clean(ecg_filtered, sampling_rate=sampling_new)
    peak_indices = obtener_picos(ecg_cleaned, sampling_new)
    if len(peak_indices) < 2:
        return H

    selected_windows = [ecg_cleaned[max(0, p - k1):min(len(ecg_cleaned), p + k2)] for p in peak_indices[1:-1]]
    if not selected_windows:
        return H

    median_window = np.median(selected_windows, axis=0)
    ar = np.array(filtrar_qrs(selected_windows, median_window))
    if ar.size == 0:
        return H

    ar = ar[:, ~np.isnan(ar).any(axis=0)]
    if ar.shape[0] == 0:
        return H

    ar -= np.mean(ar, axis=0)
    columnas, filas = ar.shape
    swd = np.zeros((wNscales, columnas, filas))
    gain = 1. / np.sqrt(wScales[:, np.newaxis])
    wName = DiscreteContinuousWaveletEx('db6')
    for j in range(filas):
        coef, _ = pywt.cwt(ar[:, j], wScales, wName)
        swd[:, :, j] = gain * coef

    Ed = np.sum(np.abs(swd) ** 2, axis=1)
    Et = np.sum(Ed, axis=0)
    if np.all(Et == 0):
        return H

    pd = Ed / Et
    Wd = -np.sum(pd * np.log(pd + np.finfo(float).eps), axis=0)
    Hd = Wd / np.log(wNscales)
    Q0 = -2 / (((wNscales + 1) / wNscales) * np.log(wNscales + 1) - 2 * np.log(2 * wNscales) + np.log(wNscales))
    Dd = Q0 * JSDiv(pd, 1 / wNscales)
    Cd = Hd * Dd

    return {'mHd': np.mean(Hd), 'mCd': np.mean(Cd)}

# Entrenamiento optimizado y sencillo
def train_model(data_folder, model_folder, verbose=False):
    records = find_records(data_folder)
    features, labels = [], []

    for record_name in records:
        record = os.path.join(data_folder, record_name)
        feat = extract_features(record)
        label = load_label(record)
        if feat is not None and not np.isnan(feat).any():
            features.append(feat)
            labels.append(label)

    if not features:
        raise RuntimeError("No hay suficientes registros válidos.")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)

    os.makedirs(model_folder, exist_ok=True)
    joblib.dump({'model': model}, os.path.join(model_folder, 'model.sav'))

# Extracción simplificada de características
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)
    sex_encoded = [sex == 'Female', sex == 'Male', sex not in ['Female', 'Male']]

    signal, fields = load_signals(record)
    sampling_signal = fields['fs']
    entropy = entropiaWavelet(signal[:, 0], sampling=sampling_signal)
    if entropy is None or np.isnan(list(entropy.values())).any():
        return None

    return np.array([age, *sex_encoded, entropy['mHd'], entropy['mCd']])


def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    model = model['model']
    features = extract_features(record)

    if features is None or np.isnan(features).any():
        if verbose:
            print(f"[WARNING] Características inválidas para: {record}")
        return float('nan'), float('nan')

    features = features.reshape(1, -1)
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output



# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)