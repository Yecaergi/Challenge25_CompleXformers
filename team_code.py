from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os
from entropiaWavelet import entropiaWavelet
from helper_code import *

def train_model(data_folder, model_folder, verbose):
    records = find_records(data_folder)
    
    if not records:
        raise FileNotFoundError("No se encontraron registros.")

    features, labels = [], []
    discarded = 0  # Contador de registros descartados

    for record_name in records:
        record = os.path.join(data_folder, record_name)
        try:
            feat = extract_features(record)
            label = load_label(record)

            # ✅ FILTRO: Solo guardar registros válidos
            if feat is not None and not np.isnan(feat).any():
                features.append(feat)
                labels.append(label)
            else:
                discarded += 1
                # ❌ SE ELIMINA EL print() PARA QUE NO IMPRIMA NADA
                # print(f"⚠️  Registro {record_name} descartado por datos inválidos.")

        except Exception:
            discarded += 1
            # ❌ SE ELIMINA EL print() PARA QUE NO IMPRIMA NADA
            # print(f"⚠️  Error al procesar {record_name}: {e}")

    # ✅ Verifica que haya datos antes de entrenar
    if not features:
        raise RuntimeError("🚨 No hay suficientes registros válidos para entrenar el modelo.")

    # ❌ SE ELIMINAN prints PARA QUE NO MUESTRE CANTIDAD DE REGISTROS
    # print(f"✅ Registros válidos: {len(features)}")
    # print(f"⚠️ Registros descartados: {discarded}")

    # Entrenar el modelo con datos filtrados
    model = RandomForestClassifier(n_estimators=12, max_leaf_nodes=34, random_state=56)
    model.fit(np.array(features), np.array(labels))
    
    # Guardar el modelo
    os.makedirs(model_folder, exist_ok=True)
    joblib.dump({'model': model}, os.path.join(model_folder, 'model.sav'))

def run_model(record, model, verbose):
    try:
        features = extract_features(record)
        if features is None or np.isnan(features).any():
            return False, 0  # ❌ NO IMPRIME ADVERTENCIA

        features = np.array(features).reshape(1, -1)
        binary_output = model.predict(features)[0]
        probability_output = model.predict_proba(features)[0][1]
        return binary_output, probability_output

    except Exception:
        return False, 0  # ❌ NO IMPRIME ERROR

def extract_features(record):
    try:
        header = load_header(record)
        age = get_age(header)
        sex = get_sex(header)

        one_hot_encoding_sex = np.zeros(3, dtype=bool)
        one_hot_encoding_sex[{'Female': 0, 'Male': 1}.get(sex, 2)] = 1

        signal, fields = load_signals(record)
        sampling_signal = fields['fs']

        # ✅ FILTRO: Si la señal es demasiado corta, se ignora
        if signal.shape[0] < sampling_signal:
            return None  # ❌ NO IMPRIME NADA

        entropy_features = entropiaWavelet(signal[:, 0], sampling=sampling_signal)
        if entropy_features is None:
            return None

        mHd = entropy_features.get('mHd', np.nan)
        mCd = entropy_features.get('mCd', np.nan)

        features = np.concatenate(([age], one_hot_encoding_sex, [mHd, mCd])).astype(np.float32)

        # ✅ FILTRO: Si las características contienen NaN, se descarta el registro
        if np.isnan(features).any():
            return None  # ❌ NO IMPRIME ADVERTENCIA

        return features

    except Exception:
        return None  # ❌ NO IMPRIME ERROR
    
# Cargar los modelos entrenados.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    
    try:
        model = joblib.load(model_filename)
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo: {e}")
        
    return model

def save_model(model_folder, model):
    filename = os.path.join(model_folder, 'model.sav')
    try:
        joblib.dump({'model': model}, filename, protocol=0)
    except Exception:
        pass  # ❌ NO IMPRIME ERROR
