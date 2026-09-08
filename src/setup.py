#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import json
import numpy as np


CONFIG_PATH = "./config/config.json"

#============================================================================================================================#
#-------------------------------------------------------- FUNCTIONS ---------------------------------------------------------#
#============================================================================================================================#
def UTAH_paths(nb_sample : int = 1) -> list:

    """
    setup paths to the datas files for UTAH data base 

    Parameters:
    ----------
    nb_sample : number of samples to load for each year (default : 1)
    """

    #---------------------------------------------
    files = []
    start_idx = 6

    for year in UTAH_FILES_YEAR:
        for i in range(start_idx, start_idx+nb_sample):

            sample_index    = str(i) if i > 9 else "0" + str(i)
            sample_name     = UTAH_FILES_SUFIX + year + sample_index + UTAH_FILES_EXTENSION
            sample_path     = os.path.join(DATAS_DIR, sample_name) 

            files.append(sample_path)

    return files

#================================================================================#
def get_config(config_path: str = CONFIG_PATH) -> dict:

    """
    Load the configuration parameters from a JSON file.

    Parameters
    ----------
    config_path : path to the configuration JSON file.
    
    Returns
    -------
    dict : dictionary containing the configuration parameters.
    """

    #---------------------------------------------
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    return config_data

#================================================================================#
def set_config(config_data : dict) -> None :

    """
    set the global parameters with the data from the configuration file

    Parameters
    ----------
    config_data : dictionary containing the configuration parameters.
    """

    #---------------------------------------------
    global EPOCHS, BATCH_SIZE, LEARNING_RATE, LOSS_FUNCTION, OPTIMIZER
    global MODEL_NAME

    model           = config_data.get("model", {})
    hyperparameters = config_data.get("hyperparameters", {})

    #------------------------------
    MODEL_NAME      = model.get("name", "default_model")

    EPOCHS          = hyperparameters.get("epochs", 10)
    BATCH_SIZE      = hyperparameters.get("batch_size", 32)
    LEARNING_RATE   = hyperparameters.get("learning_rate", 0.001)
    LOSS_FUNCTION   = hyperparameters.get("loss_function", "mse")
    OPTIMIZER       = hyperparameters.get("optimizer", "adam")

#================================================================================#
def print_hyperparameters() -> None :

    """
    print configuration data from the configuration file
    """

    #---------------------------------------------
    print("\n#----------- Configuration ------------#")

    print(f"Epochs          : {EPOCHS}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"learning rate   : {LEARNING_RATE}")
    print(f"loss function   : {LOSS_FUNCTION}")
    print(f"optimizer       : {OPTIMIZER}")

    print("#--------------------------------------#\n")


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
DATAS_DIR   = "./data/"
MODELS_DIR  = "./models/"

UTAH_FILES_SUFIX        = "measurements_"
UTAH_FILES_YEAR         = ["2018_", "2022_"]
UTAH_FILES_SAMPLE       = 5
UTAH_FILES_EXTENSION    = ".pickle"

UTAH_FILES = []

EPOCHS          = 10
BATCH_SIZE      = 16
LEARNING_RATE   = 0.001

MODEL           = "CAE"
MODEL_NAME      = "default_model"