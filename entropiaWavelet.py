import numpy as np
import pywt  
from JSDiv import JSDiv
import matplotlib.pyplot as plt
from scipy.signal import resample, medfilt
import neurokit2 as nk
import scipy.signal as sig
import concurrent.futures
from scipy.signal import butter, freqz, filtfilt
from helper_code import *

# Excepción personalizada para procesamiento de señales
class SignalProcessingException(Exception):
    pass

# Parámetros fijos
dd, uu, ma_st = 30, 10, 2
sampling_new = 1000  
k1, k2 = 48, 77  

# Función optimizada para detección de picos R
def obtener_picos(ecg_normalized, sampling_new):
    if len(ecg_normalized) < 50:  # Señal mínima para procesar
        return np.array([])
    
    try:
        _, rpeaks = nk.ecg_peaks(ecg_normalized, sampling_rate=sampling_new)
        return rpeaks['ECG_R_Peaks']
    except Exception as e:
        print(f"Error en obtener_picos: {e}")
        return np.array([])
    


# Función optimizada para el filtro de mediana con manejo de errores
def filtro_mediana(ecg):
    try:
        kernel_size1 = min(201, len(ecg))  
        kernel_size2 = min(601, len(ecg))  
        
        estblwander = medfilt(ecg, kernel_size1)
        estblwander2 = medfilt(estblwander, kernel_size2)
        
        return ecg - estblwander2  
    except Exception as e:
        print(f"Error en filtro_mediana: {e}")
        return ecg  

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
    
    # Paralelización de procesamiento por bloques de datos
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for jj in range(0, NN, NN // 4):
            futures.append(executor.submit(promediador_rt, xx[jj:jj + NN // 4], DD, UU, xx_ci, yy_ci, kk_offset=jj))
        
        # Obtener los resultados de las tareas en paralelo
        for jj, future in enumerate(futures):
            yy_aux, xx_ci, yy_ci = future.result()
            yy[jj * (NN // 4): (jj + 1) * (NN // 4)] = yy_aux
    
    for ii in range(1, MA_stages):
        xx_ci, yy_ci = promediador_rt_init(yy, DD, UU)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for jj in range(0, NN, NN // 4):
                futures.append(executor.submit(promediador_rt, yy[jj:jj + NN // 4], DD, UU, xx_ci, yy_ci, kk_offset=jj))

            for jj, future in enumerate(futures):
                yy_aux, xx_ci, yy_ci = future.result()
                yy[jj * (NN // 4): (jj + 1) * (NN // 4)] = yy_aux
    
    xx_aux = np.roll(xx, int((DD - 1) / 2 * MA_stages * UU))
    yy = xx_aux - yy
    
    return yy

def filtrar_qrs(ventanas, median_window, threshold=0.98):
    return [
        window for window in ventanas
        if np.correlate(window.flatten(), median_window, mode='valid') / (np.linalg.norm(window) * np.linalg.norm(median_window)) > threshold
    ]

wNscales = 16
wMaxScale=16
wScales = np.arange(1, wMaxScale + 1)
# Cálculo de entropía wavelet con manejo de errores
def entropiaWavelet(signal_data, sampling):
    try:
        # Diccionario de salida con valores predeterminados
        H = {'mHd': np.nan, 'mCd': np.nan}

        # Verificar si la señal es demasiado corta o contiene NaN
        if signal_data is None or len(signal_data) < sampling or np.isnan(signal_data).any():
            raise SignalProcessingException("Señal demasiado corta para procesar o contiene NaN.")

        # Remuestreo de la señal
        new_length = int(len(signal_data) * (sampling_new / sampling))
        ecg = resample(signal_data, new_length)

        # Intentar filtrado de Lyons, si falla, usar mediana
        try:
            ecg_filtered = filtro_peine_DCyArmonicas(ecg, DD=dd, UU=uu, MA_stages=ma_st)
            if np.all(ecg_filtered == 0):
                raise SignalProcessingException("Filtro de Lyons falló.")
        except:
            ecg_filtered = filtro_mediana(ecg)

        if len(ecg_filtered) < 10:
            raise SignalProcessingException("Señal demasiado corta para limpieza.")
        
        # Limpieza con NeuroKit
        ecg_cleaned = nk.ecg_clean(ecg_filtered, sampling_rate=sampling_new)

        # Detección de picos R
        peak_indices = obtener_picos(ecg_cleaned, sampling_new)
        if len(peak_indices) < 2:
            return H  # Devuelve los valores NaN

        # Crear ventanas alrededor de los picos seleccionados
        selected_windows = [
            ecg_cleaned[max(0, peak - k1):min(len(ecg_cleaned), peak + k2)]
            for peak in peak_indices[1:-1]
        ]

        if len(selected_windows) == 0:
            return H  # Devuelve NaN

        # Calcular la mediana de todas las ventanas seleccionadas
        median_window = np.median(selected_windows, axis=0)

        # Filtrar las ventanas QRS usando la correlación
        ar = np.array(filtrar_qrs(selected_windows, median_window))

        # Verificar que `ar` no esté vacío
        if ar.size == 0:
            return H  # Devuelve NaN
        
        # Eliminar columnas con NaN
        ar = ar[:, ~np.isnan(ar).any(axis=0)]
        
        #plt.figure()
        #for xx in ar:
         #   plt.plot(xx)
        # Asegurar que aún quedan datos después de la limpieza
        if ar.shape[0] == 0:
            return H  # Devuelve NaN
        
        # Quitar la línea base a cada onda
        ar = ar - np.mean(ar, axis=0)

        # Calcular la Transformada Wavelet
        columnas, filas = ar.shape
        swd = np.zeros((wNscales, columnas, filas))
        wName = DiscreteContinuousWaveletEx('db6')

        gain = 1. / np.sqrt(wScales[:, np.newaxis])
        for j in range(filas):
            coef, _ = pywt.cwt(ar[:, j], wScales, wName)
            swd[:, :, j] = gain * coef

        # Cálculo de la entropía
        Ed = np.squeeze(np.sum(np.abs(swd) ** 2, axis=1))
        Et = np.sum(Ed, axis=0)

        if np.all(Et == 0):
            return H  # Devuelve NaN

        pd = Ed / (np.ones((wNscales, 1)) * Et)

        # Entropía total por latidos
        Wd = - np.sum(pd * np.log(pd + np.finfo(float).eps), axis=0)
        Hd = Wd / np.log(wNscales + np.finfo(float).eps)

        # Constante de normalización
        Q0 = -2 / (((wNscales + 1) / wNscales) * np.log(wNscales + 1) - 2 * np.log(2 * wNscales) + np.log(wNscales))
        
        # Desequilibrio de Jensen-Shannon por latidos
        Dd = Q0 * JSDiv(pd, 1 / wNscales)
        Cd = Hd * Dd

        return {'mHd': np.mean(Hd), 'mCd': np.mean(Cd)}
    
    except Exception as e:
        print(f"Error en entropiaWavelet: {e}")
        return H  # Devuelve valores NaN en caso de error


# Truco para incorporar las Db6 en pywt
ex = pywt.DiscreteContinuousWavelet('db6')
class DiscreteContinuousWaveletEx(type(ex)):
   def __init__(self, name=u'', filter_bank=None):
       super(type(ex), self)
       pywt.DiscreteContinuousWavelet.__init__(self, name, filter_bank)
       self.complex_cwt = False
