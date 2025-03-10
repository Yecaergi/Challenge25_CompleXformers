from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os
from entropiaWavelet import entropiaWavelet
from helper_code import *  # Si tienes funciones adicionales en helper_code.py

def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)
   
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data using parallel processing
    if verbose:
        print('Extracting features and labels from the data...')

    # Paralelización de la extracción de características
    def extract_feature_for_record(record):
        features = extract_features(record)
        if features is None:
            return None, None  # Devuelve None si no se puede extraer características
        label = load_label(record)
        return features, label
    
    # Paralelización de la iteración sobre registros usando joblib.Parallel
    if verbose:
        print("Extracting features in parallel...")
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(extract_feature_for_record)(os.path.join(data_folder, record)) for record in records)

    # Filtrar los resultados para eliminar cualquier `None` (características no válidas)
    results = [result for result in results if result[0] is not None]

    # Verificar si hay registros válidos
    if len(results) == 0:
        raise ValueError("No valid features were extracted from the records.")

    # Desempaquetar los resultados
    features, labels = zip(*results)
    features = np.array(features)
    labels = np.array(labels)

    # Comprobar si todas las características tienen la misma longitud
    if len(set(len(f) for f in features)) > 1:
        raise ValueError("The extracted features have different lengths.")
    
    # Estratificación de los datos: Dividir en entrenamiento y prueba manteniendo la proporción de clases
    if verbose:
        print("Stratifying and splitting the data into training and testing sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=56, stratify=labels
    )

    # Verificar la distribución de clases después de la división
    if verbose:
        print("Distribution of classes in the training set:")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        print(dict(zip(unique_train, counts_train)))
        
        print("Distribution of classes in the test set:")
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        print(dict(zip(unique_test, counts_test)))

    # Train the model
    if verbose:
        print('Training the model on the data...')
    
    # Define the parameters for the random forest classifier
    n_estimators = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state = 56  # Random state; set for reproducibility.

    # Fit the model (con soporte para paralelo en RandomForestClassifier)
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_leaf_nodes=max_leaf_nodes, 
        random_state=random_state, 
        n_jobs=-1  # Utilizar todos los núcleos de CPU disponibles
    ).fit(X_train, y_train)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Model training and saving complete.')
        print()



# Load your trained models.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model.
def run_model(record, model, verbose):
    if verbose:
        print("Cargando el modelo...")

    # Load the model.
    model = model['model']

    # Extract the features.
    features = extract_features(record)
    if features is None:
        return None, None
    features = features.reshape(1, -1)  # Asegúrate de que tenga la forma adecuada

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

# Extract your features.
def extract_features(record):
    try:
        header = load_header(record)
        age = get_age(header)
        sex = get_sex(header)
    
        one_hot_encoding_sex = np.zeros(3, dtype=bool)
        if sex == 'Female':
            one_hot_encoding_sex[0] = 1
        elif sex == 'Male':
            one_hot_encoding_sex[1] = 1
        else:
            one_hot_encoding_sex[2] = 1

        signal, fields = load_signals(record)
        sampling_signal = fields['fs']
        
        mHd_values = []
        mCd_values = []
      
        # Iterar sobre todas las señales en signal[:, i]
        for i in range(signal.shape[1]):
            qrs = signal[:, i]  # Tomar la señal en la columna i
            entropy_features = entropiaWavelet(qrs, sampling=sampling_signal)  # Calcular las características de entropía
    
            mHd = entropy_features['mHd']  # Extraer mHd
            mCd = entropy_features['mCd']  # Extraer mCd
    
            mHd_values.append(mHd)  # Almacenar mHd
            mCd_values.append(mCd)  # Almacenar mCd

        # Convertir a arrays de numpy (opcional, dependiendo de cómo desees usar los resultados)
        mHd_values = np.array(mHd_values)
        mCd_values = np.array(mCd_values)

        HML = np.sqrt(np.nanmean(mHd_values**2))
        CML = np.sqrt(np.nanmean(mCd_values**2))

        features = np.concatenate(([age], one_hot_encoding_sex, [HML, CML]))
        return np.asarray(features, dtype=np.float32)
    
    except Exception as e:
        print(f'Error extrayendo características de {record}: {e}')
        return None

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

