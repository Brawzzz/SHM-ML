#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import json
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

#================================================================================#
def explore_file(file_path: str) -> dict:

    """
    explore file content based on the file extension.
    Supported formats: (.csv, .xlsx, .pickle, .pkl, .json)
    
    :params file_path: path to the file to explore

    :return datas: dictionnary containing: 'format', 'data' and 'metadata'
    """

    #---------------------------------------------
    if not os.path.exists(file_path):
        print(f"No such file or directory: '{file_path}'")
        return datas

    #---------------------------------------------
    datas = {
        'format': None,
        'data': None, 
        'metadata': {}
    }

    extension       = os.path.splitext(file_path)[1].lower()
    datas['format'] = extension

    #---------------------------------------------
    try:

        #-------------------------
        if extension == '.csv':

            datas['data']       = pd.read_csv(file_path)
            datas['metadata']   = {'source': 'csv'}

        #-------------------------
        elif extension in ['.xlsx', '.xls']:

            xls       = pd.ExcelFile(file_path)
            tab_0     = xls.sheet_names[0]

            datas['data']       = pd.read_excel(file_path, sheet_name=tab_0)
            datas['metadata']   = {'sheet_names': xls.sheet_names, 'extracted_sheet': tab_0}

        #-------------------------
        elif extension in ['.pickle', '.pkl']:

            with open(file_path, 'rb') as f:
                raw_data = pkl.load(f)

            #---------------
            datas['data']   = raw_data
            metadata        = {'type': type(raw_data).__name__}

            #---------------
            if isinstance(raw_data, pd.DataFrame):
                metadata['shape'] = raw_data.shape

            #---------------
            elif isinstance(raw_data, dict):
                metadata['keys'] = list(raw_data.keys())

            #---------------
            elif hasattr(raw_data, 'shape'):
                metadata['shape'] = raw_data.shape
                
            datas['metadata'] = metadata

        #------------------------- 
        elif extension == '.json':

            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            #---------------
            if isinstance(raw_data, (dict, list)):

                try:
                    df = pd.DataFrame(raw_data)

                    datas['data']       = df
                    datas['metadata']   = {'type': 'DataFrame', 'shape': df.shape}

                except ValueError:
            
                    datas['data']       = raw_data
                    datas['metadata']   = {'type': type(raw_data).__name__}

        #-------------------------
        else:
            print(f"extension not supported : '{extension}'")
            return datas

        return datas

    #---------------------------------------------
    except Exception as e:
        print(f"Error while processing file : {e}")
        return datas

#================================================================================#
def display_data(datas : dict) -> None:

    """
    display theb datas and the metadatas collected from explore_file() function
    
    :params datas: dictionnart containing the datas, obtained with explore_file() function
    """

    #---------------------------------------------  
    if not datas or datas.get('format') is None:
        print("Erreur : Le résultat fourni est vide ou invalide.")
        return

    #--------------------------------------------- 
    print("\n" + "="*60)
    print(f"EXPLORATION (Format : {datas.get('format').upper()})")
    print("="*60)

    #--------------------------------------------- 
    print("\nMETADATAS :")
    metadata = datas.get('metadata', {})

    #---------------
    if not metadata:
        print("   (Aucune métadonnée disponible)")
    else:
        for key, value in metadata.items():
            print(f"   {str(key).capitalize():<15} : {value}")

    #---------------
    print("\nDATAS :")
    data = datas.get('data')

    if data is None:
        print("   (Aucune donnée brute extraite)")
    
    #---------------
    elif isinstance(data, pd.DataFrame):

        print(f"   Type : Pandas DataFrame | Dimensions : {data.shape[0]} lignes x {data.shape[1]} cols")
        print("   Aperçu des 5 premières lignes :")
        print("-" * 60)
        print(data.head(5))
        print("-" * 60)
        
    #---------------
    elif isinstance(data, dict):

        print(f"   Type : Dictionary | Keys : {len(data)}")

        for i, (k, v) in enumerate(data.items()):

            if i >= 10:
                print("      ...")
                break
                
            if hasattr(v, 'shape'):
                print(f"      - '{k}' \t\t: {type(v).__name__} (shape : {v.shape})")
            elif isinstance(v, list):
                print(f"      - '{k}' \t\t: list (length: {len(v)})")
            elif isinstance(v, dict):
                print(f"      - '{k}' \t\t: dict (keys: {len(v.keys())})")
            else:
                print(f"      - '{k}' \t\t: {type(v).__name__}")
                
    #---------------
    elif hasattr(data, 'shape'):

        print(f"   Type : {type(data).__name__} | Dimensions : {data.shape}")

        try:
            print(data[:min(3, len(data))])
        except Exception:
            print("   (error while display overview)")
            
    #---------------
    elif isinstance(data, list):

        print(f"   Type : List | Length : {len(data)}")

        for i, item in enumerate(data[:3]):
            extract = str(item)[:100] + "..." if len(str(item)) > 100 else str(item)
            print(f"      [{i}] {type(item).__name__} : {extract}")
            
    #---------------
    else:

        print(f"   Type : {type(data).__name__}")

        repr_data = str(data)
        if len(repr_data) > 200:
            print(f"   Valeur : {repr_data[:200]}... (tronqué)")
        else:
            print(f"   Valeur : {repr_data}")
            
    print("="*60 + "\n")

#================================================================================#
def valid_signals(df, ref_size) -> np.ndarray:

    """
    Extract valid signals from a DataFrame based on a 
    reference size and return the first valid signal found.

    :param df:          DataFrame containing the signals.
    :param ref_size:    Reference size for the signals.
    """

    #---------------------------------------------
    found_signals = []
    
    for col in df.columns:

        serie = pd.to_numeric(df[col].iloc[1:], errors='coerce').dropna().values
    
        if(len(serie) >= ref_size and min(serie) < 0):
            found_signals.append(serie[:ref_size])

    #---------------
    if len(found_signals) == 3:
        return(found_signals[1])

    elif len(found_signals) > 0:
        return (found_signals[0])
        
    return None 

#================================================================================#
def build_dataset():

    signals_X   = []
    labels_y    = [] 
    
    #---------------- Baseline data extraction (Crackless) ----------------#
    baseline_file = "./data/cracks/Crackless.xlsx"

    if os.path.exists(baseline_file):

        print(f"Processing {baseline_file}")

        df_baseline     = pd.read_excel(baseline_file, sheet_name=0)
        baseline_amp    = pd.to_numeric(df_baseline.iloc[1:, 1], errors='coerce').dropna().values

        REF_SIZE = len(baseline_amp)

        signals_X.append(baseline_amp)
        labels_y.append([0.0, 0.0])
    
    #------------------------ Crack data extraction -----------------------#
    crack_depths = {"0.5mm": 0.5, "0.8mm": 0.8, "1mm": 1.0, "1.6mm": 1.6}
    crack_files  = {"./data/cracks/L10.xlsx": 10.0, 
                    "./data/cracks/L20.xlsx": 20.0,
                    "./data/cracks/L30.xlsx": 30.0,
                    "./data/cracks/L50.xlsx": 50.0,
                    "./data/cracks/L70.xlsx": 70.0,}

    #---------------------------------------------
    for file, length in crack_files.items():

        if os.path.exists(file):

            print(f"Processing {file}")
            
            for tab, deepth in crack_depths.items():

                #---------------
                try:

                    df_crack    = pd.read_excel(file, sheet_name=tab)
                    crack_amp   = valid_signals(df_crack, REF_SIZE)

                    #---------------
                    if crack_amp is not None:

                        signals_X.append(crack_amp)
                        labels_y.append([length, deepth])

                    else:
                        print(f"Failed {tab} no signals found")
                
                except ValueError:
                    print(f"Tab {tab} not found in {file}")
                    
    #---------------------------------------------
    X = np.array(signals_X)
    y = np.array(labels_y)
    
    print("\nDataset construction completed")
    print(f"-> Forme de X (Signaux) : {X.shape} (Expériences x Échantillons temporels)")
    print(f"-> Forme de y (Labels)  : {y.shape} (Expériences x [Longueur, Profondeur])")
    
    return X, y


#================================================================================#
def display_datasets(X : np.ndarray, y : np.ndarray):

    """
    Display the dataset signals with their corresponding labels.
    """

    #---------------------------------------------
    plt.figure(figsize=(12, 6))

    plt.plot(X[0], label='Plaque Saine (0mm)', color='green', linewidth=2)

    #---------------
    idx_l10 = np.where((y[:, 0] == 10.0) & (y[:, 1] == 1.6))[0]
    if len(idx_l10) > 0:
        plt.plot(X[idx_l10[0]], label='Fissure 10mm (1.6mm prof.)', color='orange', linestyle='--')

    idx_l20 = np.where((y[:, 0] == 20.0) & (y[:, 1] == 1.6))[0]
    if len(idx_l20) > 0:
        plt.plot(X[idx_l20[0]], label='Fissure 20mm (1.6mm prof.)', color='red', linestyle='-.')

    #---------------
    plt.title("Visualisation du Dataset : Comparaison des signaux (Trajet MID)", fontsize=14)
    plt.xlabel("Échantillons (Temps)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.show()

#================================================================================#
def baseline_data(baseline_file="./data/Crackless.xlsx"):

    """
    healty data extrction for AE training.

    :params baseline_file: path to the baseline data file
    """

    #---------------------------------------------------------
    if not os.path.exists(baseline_file):

        print(f"ERROR : {baseline_file} no such file or directory")
        return(None, None)

    #---------------------------------------------------------
    healthy_signals = []
    df_healty       = pd.read_excel(baseline_file, sheet_name=0)
    cols_idx        = [1, 3, 5] 
    
    for col in cols_idx:
        
        signal = pd.to_numeric(df_healty.iloc[1:, col], errors='coerce').dropna().values
        healthy_signals.append(signal)
        
    ref_size    = len(healthy_signals[0])
    signals     = [sig[:ref_size] for sig in healthy_signals if len(sig) >= ref_size]
    X_train     = np.array(signals)

    #---------------------------------------------------------
    scaler              = MinMaxScaler()
    X_train_normalise   = scaler.fit_transform(X_train)
    
    return(X_train_normalise, scaler)

#================================================================================#
def cracks_data(crack_file, scaler, ref_size=668):

    """
    cracks signal extraction and normalisation.

    :params crack_file: path tho the cracks datas file
    :params scaler:     scaler used for baselin data normalisation
    :ref_size:          reference size for valid signals

    :retrun X_test_normalise: noramlise cracks data
    :return signals_datas: information on the valid signals 
    """

    #---------------------------------------------------------
    cracks_signals  = []
    signals_datas   = []
    tabs            = ["0.5mm", "0.8mm", "1mm", "1.6mm"]

    #---------------------------------------------------------
    for file in crack_file:

        #-------------------------
        if not os.path.exists(file):
            print(f"No such file or directory : {file}")
            continue

        #-------------------------
        for tab_i in tabs:
            
            try:

                df  = pd.read_excel(file, sheet_name=tab_i)

                valid_signal = None

                #-------------------------
                for col in df.columns[1:]: 
                    
                    serie = pd.to_numeric(df[col].iloc[1:], errors='coerce').dropna().values

                    if len(serie) >= ref_size - 20 and min(serie) < 0:
                        
                        #---------------
                        if len(serie) < ref_size:
                            lack    = ref_size - len(serie)
                            serie   = np.pad(serie, (0, lack), 'constant')
                            
                        #---------------
                        valid_signal = serie[:ref_size]
                        break

                #-------------------------
                if valid_signal is not None:

                    cracks_signals.append(valid_signal)
                    signals_datas.append(f"{file} - Depth {tab_i}")

                #-------------------------
                else:
                    print(f" no valid signals in {file} ({tab_i})")

            except ValueError:
                print(f"  ⚠️ Onglet {tab_i} introuvable dans {file}")
                
    #---------------------------------------------------------
    X_test = np.array(cracks_signals)
    
    if len(X_test) == 0:
        print(f"no cracks found")
        return None, None

    X_test_normalise = scaler.transform(X_test)
    
    return(X_test_normalise, signals_datas)