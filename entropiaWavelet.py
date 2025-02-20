# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:41:40 2025

@author: GVC
"""
import numpy as np
import pywt  
from JSDiv import JSDiv
import matplotlib.pyplot as plt
from scipy.signal import resample
import neurokit2 as nk
import wfdb
import os


# Funciones para el filtro de Lyons
def promediador_rt_init(xx, DD, UU):
    hh_u = np.zeros(DD * UU)
    hh_u[::UU] = 1
    hh_u = np.flip(hh_u)   
    yy_ci = np.zeros(UU)
    yy_ci[:] = np.sum(xx[:(DD * UU)] * hh_u)
    xx_ci = xx[:(DD * UU)]

    return xx_ci, yy_ci

def promediador_rt(xx, DD, UU, xx_ci, yy_ci, kk_offset=0):
    NN = xx.shape[0]
    yy = np.zeros_like(xx)

    if kk_offset == 0:
        for kk in range(UU):
            yy[kk] = xx[kk] - xx_ci[kk] + yy_ci[kk]
        yy[kk:DD * UU] = yy[kk]
        hh_u = np.zeros(DD * UU)
        hh_u[::UU] = 1
        hh_u = np.flip(hh_u)
        for kk in range(DD * UU, (DD * UU) + UU):
            ii = kk-1
            yy[ii] = np.sum(xx[kk - (DD * UU):kk] * hh_u)

    else:
        for kk in range(UU):
            yy[kk] = xx[kk] - xx_ci[kk] + yy_ci[kk]
        for kk in range(UU, DD * UU):
            yy[kk] = xx[kk] - xx_ci[kk] + yy[kk - UU]
        
        kk += 1
    for kk in range(kk, NN):
        yy[kk] = xx[kk] - xx[kk - DD * UU] + yy[kk - UU]

    xx_ci = xx[(NN - DD * UU):]
    yy_ci = yy[(NN - UU):]

    return yy.copy() / DD, xx_ci.copy(), yy_ci.copy()

def filtro_peine_DCyArmonicas(xx, DD=16, UU=2, MA_stages=2):
    NN = len(xx)
    xx_ci, yy_ci = promediador_rt_init(xx, DD, UU)
    yy = np.zeros_like(xx)
    for jj in range(0, NN, NN // 4):
        yy_aux, xx_ci, yy_ci = promediador_rt(xx[jj:jj + NN // 4], DD, UU, xx_ci, yy_ci, kk_offset=jj)
        yy[jj:jj + NN // 4] = yy_aux
    for ii in range(1, MA_stages):
        xx_ci, yy_ci = promediador_rt_init(yy, DD, UU)
        for jj in range(0, NN, NN // 4):
            yy_aux, xx_ci, yy_ci = promediador_rt(yy[jj:jj + NN // 4], DD, UU, xx_ci, yy_ci, kk_offset=jj)
            yy[jj:jj + NN // 4] = yy_aux
    xx_aux = np.roll(xx, int((DD - 1) / 2 * MA_stages * UU))
    yy = xx_aux - yy
    
    return yy

def obtener_picos(ecg_normalized):
    _, rpeaks = nk.ecg_peaks(ecg_normalized, sampling_rate=fs)
    
    return rpeaks['ECG_R_Peaks']

# Filtrar las ventanas con correlación mayor al umbral  98%
def filtrar_qrs(ventanas, median_window, threshold=0.98):
    
    return [
        window for window in ventanas
        if np.correlate(window.flatten(), median_window, mode='valid') / (np.linalg.norm(window) * np.linalg.norm(median_window)) > threshold
    ]

#Truco para incorporar las Db6 en pywt
ex = pywt.DiscreteContinuousWavelet('db6')
class DiscreteContinuousWaveletEx(type(ex)):
   def __init__(self, name=u'', filter_bank=None):
       super(type(ex), self)
       pywt.DiscreteContinuousWavelet.__init__(self, name, filter_bank)
       self.complex_cwt = False


fs=400         
nyq_frec = fs/2
dd=30   
uu = 10  
ma_st = 2 
demora_rl = int((dd-1)/2*ma_st*uu)
k1 = 48
k2 = 77
units = 'mV'  
sampling_new = 1000  


def entropiaWavelet(signal_data, **kwargs): 
  
    wName = kwargs.get('wName', 'gaus6')
    wMaxScale = kwargs.get('wMaxScale', 16)
    wTscale = kwargs.get('wTscale', 'continua')

    
    if 'wName' in kwargs:
        wName = kwargs['wName']
    if 'wMaxScale' in kwargs:
        wMaxScale = kwargs['wMaxScale']
    if 'wTscale' in kwargs:
        wTscale = kwargs['wTscale']

    
    if wTscale == 'discreta':
        wScales = 2 ** np.arange(wMaxScale)       
    else:
        wScales = np.arange(1, wMaxScale + 1)     
    
    wNscales = len(wScales)
    
    H = {
    'pd': np.nan,    
    'mWd': np.nan,
    'mDd': np.nan,
    'mHd': np.nan,
    'sHd': np.nan,
    'mCd': np.nan,
    'sCd': np.nan,
    'mEt': np.nan,
    'filas': np.nan,
    'wName': np.nan,
    'wScales': np.nan,
    'wNscales': np.nan,
    'wMaxScale': np.nan,
    'wTscale': np.nan,
}
    
    # Calcular la nueva longitud de la señal
    new_length = int(len(signal_data) * (sampling_new / fs))

    # Realizar el re-muestreo para aplicar mismo método que en PTB (16 escalas)
    ecg = resample(signal_data, new_length)

    # Filtrado de Lyons
    ecg_filtered = filtro_peine_DCyArmonicas(ecg, DD=dd, UU=uu, MA_stages=ma_st)
    
    # Normalización y limpieza de la señal
    ecg_cleaned = nk.ecg_clean(ecg_filtered.flatten(), sampling_rate=fs)
    
    # Detección de picos R
    peak_indices = obtener_picos(ecg_cleaned)
    
    # Seleccionar solo los picos entre el segundo y el anteúltimo
    if len(peak_indices) > 2:  
        selected_peaks = peak_indices[1:-1]
    else:
        print(f'La derivación tiene menos de 3 picos R detectados. No se procesará.') 
        
    # Crear ventanas alrededor de los picos seleccionados
    selected_windows = [
        np.pad(
            ecg_cleaned[max(0, peak - k1):min(len(ecg_cleaned), peak + k2)],
            (0, max(0, len(ecg_cleaned[max(0, peak - k1):min(len(ecg_cleaned), peak + k2)]) - len(ecg_cleaned))),
            'constant'
        )
        for peak in selected_peaks
    ]

    # Calcular la mediana de todas las ventanas seleccionadas
    median_window = np.median(selected_windows, axis=0)

    # Filtrar las ventanas QRS usando la correlación
    ar = filtrar_qrs(selected_windows, median_window)
    ar = np.array(ar)
    
    hayNaN = np.nanmean(ar, axis=0)
    ar = ar[:, ~np.isnan(hayNaN)]
     
    
    #----------------------------------------------------------------------
    #  quito la linea de base a cada onda
    # ----------------------------------------------------------------------
    columnas, filas = ar.shape                
   
    ar = ar - np.ones((columnas, 1)) *np.mean(ar, axis=0)  # elimino la media de cada onda
  
    #--------------------------------------------------------------
    # obtengo la escala wavelet de las derivaciones ortogonales
    #--------------------------------------------------------------
  
    swd = np.zeros((wNscales, columnas, filas))
    wName = DiscreteContinuousWaveletEx('db6')

    if wTscale == 'discreta':
         sqrt22 = np.sqrt(2) / 2
         for j in range(filas):
             _, sd = pywt.swt(ar[:, j], wName, level=wNscales)
             gain = sqrt22 / np.sqrt(wScales[:, None])
             swd[:, :, j] = gain * np.array(sd)
    else:
         gain = 1. / np.sqrt(wScales[:, np.newaxis])
         # Calcular la CWT para cada fila
         for j in range(filas):
             coef, _ = pywt.cwt(ar[:, j], wScales, wName)
             swd[:, :, j] = gain * coef       
          
    
    H['wNscales'] = wNscales;
    H['wMaxScale'] = wMaxScale;
    H['wScales'] = wScales;
    H['wName'] = wName;
    H['wTscale'] = wTscale;        
   
     
    columnas, filas=ar.shape
   
    #----------------------------------------------------------------------
    #  Analizo la entropia completa de cada onda
    #----------------------------------------------------------------------
    
    # energia escalas x latidos
    Ed = np.squeeze(np.sum(np.abs(swd) ** 2, axis=1))

    #  energia total x filas
    Et = np.sum(Ed, axis=0)
    
    #  distribucion de probabilidad escalas x latidos
    pd = Ed / (np.ones((wNscales,1)) * Et)
  
    # entropia total x latidos
    Wd = - np.sum(pd * np.log(pd + np.finfo(float).eps), axis=0)
    # desorden x latidos (entropia normalizada)
    Hd = Wd / np.log(wNscales + np.finfo(float).eps)

    # constante de normalización
    Q0 = -2 / (((wNscales + 1) / wNscales) * np.log(wNscales + 1) - 2 * np.log(2 * wNscales) + np.log(wNscales))
 
    # desequilibrio de Jensen-Shannon x latidos
    Dd = Q0 * JSDiv(pd, 1 / wNscales)  
    
    Cd = Hd * Dd
   
    H['pd'] = np.mean(pd, axis=1, keepdims=True)
    H['mEt'] = np.mean(Et)     # energia media de los latidos
    H['mWd'] = np.mean(Wd)     # entropia media de los latidos
    H['mHd'] = np.mean(Hd)     # desorden medio (entropia normalizada) de los latidos
    H['sHd'] = np.std(Hd)      # desorden SD (entropia normalizada) de los latidos
    H['mDd'] = np.mean(Dd)     # desequilibrio medio de los latidos
    H['mCd'] = np.mean(Cd)     # complejidad media de los latidos
    H['sCd'] = np.std(Cd)      # complejidad SD de los latidos

    wdFilas = len(Wd);    
    H['filas'] = wdFilas
    
    return H

#----------------------------------------------------------------
# Veamos cómo funciona en algunos pacientes...
#----------------------------------------------------------------

#archivo_base = "C:/Users/eah/Desktop/challenge2025gi/code15_wfdb/267"

# Verificar si los archivos existen
#archivo_dat = archivo_base + '.dat'
#archivo_hea = archivo_base + '.hea'

# Comprobamos si ambos archivos están en la ruta especificada
#if os.path.exists(archivo_dat) and os.path.exists(archivo_hea):
    
#    try:
#        record = wfdb.rdrecord(archivo_base)
#        metadatos=record.__dict__
        
#    except Exception as e:
#        print(f"Error al intentar leer el archivo con wfdb: {e}")
#else:
#    print(f"Uno o ambos archivos no se encuentran en la ruta especificada: {archivo_dat}, {archivo_hea}")

# Buscar el comentario que contiene "Chagas label"
#chagas_label = None
#for comentario in metadatos['comments']:
#    if 'Chagas label' in comentario:
#        chagas_label = comentario.split(':')[1].strip()
#        break

#print(f"Chagas label: {chagas_label}")



#tamaño=record.p_signal.shape
#xx = record.p_signal[:, 0]


#plt.figure()
#plt.plot(xx)


#resultado = entropiaWavelet(xx, wName='db6', wMaxScale=16, wTscale='continua')




