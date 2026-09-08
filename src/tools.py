#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import json
import keras
import torch
import pickle
import numpy as np
import argparse as ap
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix

import setup as stp


#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def save_model(model, scaler: TransformerMixin, model_name: str = "") -> None:

    """
    Save a model (.keras for Keras, .pth for PyTorch) along with its scaler.
    
    Parameters
    ----------
    model           : the model to save (Keras or PyTorch)
    scaler          : model's scaler tool
    model_name      : name of the model 
    """

    #---------------------------------------------
    os.makedirs(stp.MODELS_DIR, exist_ok=True)

    if model_name == "":

        sufix = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        base_name = f"model_{sufix}"

    else:
        base_name = f"{model_name}"

    #------------------------------
    print(f" -> Saving model ...", end="", flush=True)

    #---------------
    if(isinstance(model, keras.Model)):
        
        model_path = os.path.join(stp.MODELS_DIR, f"{base_name}.keras")
        model.save(model_path)

    #---------------
    elif isinstance(model, torch.nn.Module):
        
        model_path = os.path.join(stp.MODELS_DIR, f"{base_name}.pth")
        torch.save(model.state_dict(), model_path)

    #---------------
    else:
        print("Error model type not recognized")

    print(f"Done ({model_path})")

    #------------------------------
    print(f" -> Saving model's scaler ...", end="", flush=True)
    
    scaler_path = os.path.join(stp.MODELS_DIR, f"{base_name}_scaler.pkl")
    
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    
    print(f"Done ({scaler_path})")


#================================================================================#
def arg_parse() -> ap.Namespace:

    """
    Get the arguments and paths from the command line.

    Returns
    ----------
    args : Object containing all parsed arguments.
    """

    #---------------------------------------------
    parser = ap.ArgumentParser(description="SHM methods for damage detection and localization")
    
    parser.add_argument("-T", "--train", type=str, default=None, 
                        help="Launch training phase with the given configuration file")

    parser.add_argument("-t", "--test", type=int, default=None, 
                        help="Launch testing phase on the corresponding signal index")
    
    parser.add_argument("-p", "--plot", type=str, default=None, 
                        help="Path to the JSON file containing training losses for plotting")
    
    args = parser.parse_args()

    #------------------------------
    if args.train is not None:
        print(f"\n#--------------- Training config file : {args.train} ---------------#\n")

    #------------------------------
    if args.test is not None:
        print(f"\n#--------------- Testing signal index : {args.test} ---------------#\n")

    #------------------------------
    if args.plot is not None:
        print(f"\n#--------------- Plotting training curve : {args.plot} ---------------#\n")

    return args

#================================================================================#
def load_model(model_path: str, 
               model_type: str, 
               model_class: type, 
               scaler_path: str, 
               model_kwargs: dict = None) -> tuple[object, TransformerMixin]:

    """
    Load a model (.keras for Keras, .pth for PyTorch) along with its scaler.

    Parameters
    ----------
    model_path : path to the saved model file.
    model_type : type of the model ("Keras" or "PyTorch")
    model_class : class of the PyTorch model (if model_type is "PyTorch")
    scaler_path : path to the saved scaler file.
    model_kwargs : dictionary of keyword arguments to initialize the PyTorch model (if model_type is "PyTorch")
    """

    #---------------------------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    #------------------------------
    if model_kwargs is None:
        model_kwargs = {}

    print(f"Loading model ...", end="", flush=True)

    if model_type == "Keras":
        from keras.models import load_model
        model = load_model(model_path)

    elif model_type == "PyTorch":
        if model_class is None:
            raise ValueError("model_class must be provided for PyTorch models.")
        
        model = model_class(**model_kwargs)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

    else:
        raise ValueError(f"Model type '{model_type}' not recognized.")

    print(f" Done ({model_path})")

    #------------------------------
    print(f"Loading scaler ...", end="", flush=True)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    print(f" Done ({scaler_path})")

    return model, scaler

#================================================================================#
def learning_curve(axes, train_losses: list) -> None:

    """
    Plot the learning curve for the training losses.

    Parameters
    ----------
    axes : matplotlib axes object to plot on
    train_losses : list of training losses for each epoch
    """

    #---------------------------------------------
    axes[0].plot(train_losses, label='Train Loss (MSE)', color='blue', linewidth=2)
    axes[0].set_title('Courbe d\'apprentissage du CAE', fontsize=14)
    axes[0].set_xlabel('Époques (Epochs)')
    axes[0].set_ylabel('Erreur de reconstruction (MSE)')
    axes[0].legend()

#================================================================================#
def confusion_matrix(axes, healthy_mse, crack_mse, threshold) -> None:

    """
    Plot the confusion matrix based on the healthy / cracked MSE values and the threshold.
    """

    #---------------------------------------------
    y_true = np.concatenate([np.zeros(len(healthy_mse)), np.ones(len(crack_mse))])
        
    all_mse = np.concatenate([healthy_mse, crack_mse])
    y_pred = (all_mse > threshold).astype(int)
    
    CM = confusion_matrix(y_true, y_pred)
        
    sns.heatmap(CM, annot=True, fmt='d', cmap='Blues', ax=axes[2], 
                xticklabels=['Sain Prédit', 'Fissure Prédite'], 
                yticklabels=['Sain Réel', 'Fissure Réelle'])
    
    axes[2].set_title('Matrice de Confusion', fontsize=14)

#================================================================================#
def error_distribution(axes, healthy_mse, crack_mse, threshold) -> None:
    
    """
    Plot the distribution of reconstruction errors for healthy and cracked signals.

    Parameters
    ----------
    axes : matplotlib axes object to plot on
    healthy_mse : numpy array of MSE for healthy signals
    crack_mse : numpy array of MSE for cracked signals
    threshold : float value of the warning threshold
    """

    #---------------------------------------------
    sns.histplot(healthy_mse, bins=50, color='green', alpha=0.6, label='Sains (Validation)', ax=axes[1], stat='density')
    sns.histplot(crack_mse, bins=50, color='red', alpha=0.6, label='Fissures (Test)', ax=axes[1], stat='density')
    
    axes[1].axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Seuil d\'alerte')
    axes[1].set_title('Séparation des erreurs de reconstruction', fontsize=14)
    axes[1].set_xlabel('Erreur MSE')
    axes[1].set_ylabel('Densité')
    axes[1].legend()


#================================================================================#
def model_perf(train_losses : list, 
               healthy_mse : np.ndarray, 
               crack_mse : np.ndarray, 
               threshold : float) -> None :

    """
    plot the model performance afetr the training phase 

    Parameters
    ----------
    train_losses : list of training losses for each epoch
    healthy_mse  : numpy array of MSE for healthy signals
    crack_mse    : numpy array of MSE for cracked signals
    threshold    : float value of the warning threshold
    """

    #---------------------------------------------
    sns.set_theme(style="whitegrid")
    (fig, axes) = plt.subplots(1, 3, figsize=(18, 5))

    #-----------------------------------
    learning_curve(axes, train_losses)
    error_distribution(axes, healthy_mse, crack_mse, threshold)
    confusion_matrix(axes, healthy_mse, crack_mse, threshold)

    #-----------------------------------
    plt.tight_layout()
    plt.show()
        

    