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
import scipy.signal as sig
import os
from scipy.signal import butter, freqz, filtfilt
from helper_code import *

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

def obtener_picos(ecg_normalized, sampling_new):
    _, rpeaks = nk.ecg_peaks(ecg_normalized, sampling_rate=sampling_new)
    
    return rpeaks['ECG_R_Peaks']

# Filtrar las ventanas con correlación mayor al umbral  98%
def filtrar_qrs(ventanas, median_window, threshold=0.98):
    
    return [
        window for window in ventanas
        if np.correlate(window.flatten(), median_window, mode='valid') / (np.linalg.norm(window) * np.linalg.norm(median_window)) > threshold
    ]

# Truco para incorporar las Db6 en pywt
ex = pywt.DiscreteContinuousWavelet('db6')
class DiscreteContinuousWaveletEx(type(ex)):
   def __init__(self, name=u'', filter_bank=None):
       super(type(ex), self)
       pywt.DiscreteContinuousWavelet.__init__(self, name, filter_bank)
       self.complex_cwt = False

dd=30         # dejarlo así
uu = 10       # dejarlo así porque resamplee a 1000Hz
ma_st = 2 
demora_rl = int((dd-1)/2*ma_st*uu)
k1 = 48
k2 = 77
units = 'mV'  
sampling_new = 1000  

class SignalProcessingException(Exception):
    """Exception raised when there are issues in signal processing."""
    pass


def entropiaWavelet(signal_data, sampling, **kwargs):
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
    
    # Verifica si la señal está vacía o tiene NaN
    if signal_data is None or len(signal_data) == 0 or np.isnan(signal_data).any():
        raise SignalProcessingException("La señal está vacía o contiene valores NaN.")

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
    new_length = int(len(signal_data) * (sampling_new / sampling))

    # Realizar el re-muestreo para aplicar el mismo método que en PTB (16 escalas)
    try:
        ecg = resample(signal_data, new_length)
    except Exception as e:
        raise SignalProcessingException(f"Error al realizar el re-muestreo de la señal: {e}")

    # Filtrado de Lyons
    try:
        ecg_filtered = filtro_peine_DCyArmonicas(ecg, DD=dd, UU=uu, MA_stages=ma_st)
    except Exception as e:
        # Si falla el filtro especializado, intentamos el filtro de mediana
        try:
            estblwander = sig.medfilt(ecg, 201)
            estblwander2 = sig.medfilt(estblwander, 601)
            ecg_filtered = ecg - estblwander2  # La señal filtrada con el filtro de mediana
        except Exception as e2:
            raise SignalProcessingException(f"Ambos filtros fallaron: {e2}")

    # Normalización y limpieza de la señal
    try:
        ecg_cleaned = nk.ecg_clean(ecg_filtered.flatten(), sampling_rate=sampling_new)
    except Exception as e:
        raise SignalProcessingException(f"Error al limpiar la señal ECG: {e}")

    # Detección de picos R
    try:
        peak_indices = obtener_picos(ecg_cleaned, sampling_new)
    except Exception as e:
        raise SignalProcessingException(f"Error al detectar los picos R: {e}")

    # Seleccionar solo los picos entre el segundo y el anteúltimo
    if len(peak_indices) > 2:  
        selected_peaks = peak_indices[1:-1]
    else:
        raise SignalProcessingException(f'La derivación tiene menos de 3 picos R detectados. No se procesará.')

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
    
    # Eliminar las columnas que contengan NaN antes de continuar
    ar = ar[:, ~np.isnan(ar).any(axis=0)]
    
    # Asegurarse de que las ventanas no sean vacías
    if ar.shape[0] == 0:
        raise SignalProcessingException("No se encontraron ventanas válidas después del filtrado. No se procesará.")
    
    # Quitar la línea base a cada onda
    columnas, filas = ar.shape
    ar = ar - np.ones((columnas, 1)) * np.mean(ar, axis=0)  # Elimino la media de cada onda

    # Analizo el ruido
    VENTANA_RUIDO = 20  # lo dejo constante a 20 μV porque resampleo a 1000 Hz
    # Calcula el ruido RMS
    rmsnoise = np.sqrt(np.mean(np.std(ar[:VENTANA_RUIDO, :], axis=0, ddof=0)**2))
    
    # Filtrar las señales cuyo ruido RMS es mayor a 20 μV
    indices_a_descartar = np.nonzero(rmsnoise > 20e-6)[0]  # 20 μV = 20e-6 V
    ar_filtrado = np.delete(ar, indices_a_descartar, axis=0)
    
    # Asegurarse de que hay datos después del filtrado
    if ar_filtrado.shape[0] == 0:
        raise SignalProcessingException("No se encontraron señales válidas después del filtrado por ruido. No se procesará.")

    # Obtengo la escala wavelet de las derivaciones ortogonales
    columnas, filas = ar_filtrado.shape
    swd = np.zeros((wNscales, columnas, filas))
    wName = DiscreteContinuousWaveletEx('db6')

    if wTscale == 'discreta':
        sqrt22 = np.sqrt(2) / 2
        for j in range(filas):
            _, sd = pywt.swt(ar_filtrado[:, j], wName, level=wNscales)
            gain = sqrt22 / np.sqrt(wScales[:, None])
            swd[:, :, j] = gain * np.array(sd)
    else:
        gain = 1. / np.sqrt(wScales[:, np.newaxis])
        # Calcular la CWT para cada fila
        for j in range(filas):
            coef, _ = pywt.cwt(ar_filtrado[:, j], wScales, wName)
            swd[:, :, j] = gain * coef

    H['wNscales'] = wNscales
    H['wMaxScale'] = wMaxScale
    H['wScales'] = wScales
    H['wName'] = wName
    H['wTscale'] = wTscale

    # Analizo la entropía completa de cada onda
    Ed = np.squeeze(np.sum(np.abs(swd) ** 2, axis=1))
    Et = np.sum(Ed, axis=0)
    
    if np.all(Et == 0):
       raise SignalProcessingException("Entropía total (Et) es 0, se saltará esta señal.")

    
    pd = Ed / (np.ones((wNscales, 1)) * Et)
  
    # Entropía total por latidos
    Wd = - np.sum(pd * np.log(pd + np.finfo(float).eps), axis=0)
    Hd = Wd / np.log(wNscales + np.finfo(float).eps)

    # Constante de normalización
    Q0 = -2 / (((wNscales + 1) / wNscales) * np.log(wNscales + 1) - 2 * np.log(2 * wNscales) + np.log(wNscales))
 
    # Desequilibrio de Jensen-Shannon por latidos
    Dd = Q0 * JSDiv(pd, 1 / wNscales)  
    Cd = Hd * Dd
   
    H['pd'] = np.mean(pd, axis=1, keepdims=True)
    H['mEt'] = np.mean(Et)     # Energía media de los latidos
    H['mWd'] = np.mean(Wd)     # Entropía media de los latidos
    H['mHd'] = np.mean(Hd)     # Desorden medio (entropía normalizada) de los latidos
    H['sHd'] = np.std(Hd)      # Desorden SD (entropía normalizada) de los latidos
    H['mDd'] = np.mean(Dd)     # Desequilibrio medio de los latidos
    H['mCd'] = np.mean(Cd)     # Complejidad media de los latidos
    H['sCd'] = np.std(Cd)      # Complejidad SD de los latidos

    wdFilas = len(Wd)    
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
 #       record = wfdb.rdrecord(archivo_base)
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
#record='C:/Users/eah/Desktop/challenge2025gi/Samitrop/samitrop_output/4991'

#signal, fields = load_signals(record)
#xx=signal[:,0]

#plt.figure()
#plt.plot(xx)
#sampling_signal=fields['fs']
#resultado = entropiaWavelet(xx,sampling_signal, wName='db6', wMaxScale=16, wTscale='continua')







