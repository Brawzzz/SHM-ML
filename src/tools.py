import os
import keras
import pickle

from datetime import datetime
from sklearn.base import TransformerMixin

import setup as stp

#================================================================================#
def save_model(model : keras.Model, scaler : TransformerMixin , model_name : str = ""):

    """
    save a model in .keras file and the scaler tool used 

    :params model:      the model to save
    :params sacler:     model's scaler tool
    :params model_name: name of the model 

    :return model_path: path to the saved model
    :return scaler_path: path to the model's scaler tools path

    """

    #------------------------------
    if(not os.path.exists(stp.MODELS_DIR)):
        os.makedirs(stp.MODELS_DIR)

    if(model_name == ""):

        sufix = (datetime.now()).strftime("%Y_%m_%d_%H_%M_%S")

        saved_model_name    = "model_" + sufix + ".keras"
        scaler_name         = "model_" + sufix + "_scaler_shm.pkl"

    else :
        saved_model_name  = f"{model_name}_shm.keras"
        scaler_name       = f"{model_name}_scaler_shm.pkl"

    #------------------------------
    model_path  = os.path.join(stp.MODELS_DIR, saved_model_name)
    scaler_path = os.path.join(stp.MODELS_DIR, scaler_name)

    model.save(model_path)

    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)

    