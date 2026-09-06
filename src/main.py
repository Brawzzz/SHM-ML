#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import json
import torch
import numpy as np 

from sklearn.model_selection import train_test_split

import AE
import CAE

import data
import tools 
import setup as stp


#============================================================================================================================#
#-------------------------------------------------------- FUNCTION ----------------------------------------------------------#
#============================================================================================================================#
def run_training(n_X_train : np.ndarray, n_X_test : np.ndarray, model : str = "CAE") -> tuple[torch.nn.Module, float, np.ndarray, np.ndarray]:

    """
    launch the training phase of a model depending on his type  

        - Auto Encodeur (AE : default) 
        - Convolutional Auto Encodeur (CAE)  

    Parameters
    ----------
    n_X_train : train dataset
    n_X_test  : test dataset
    model    : str describing the model's type (default : AE)

    Returns
    ----------
    model        : trained model 
    threshold    : anomalies threshold
    recons       : inputs reconstruction 
    train_losses : training losses 
    """

    #---------------------------------------------
    if not (len(n_X_train) > 0 and len(n_X_test) > 0):
        print(f"datasets are empty : {len(n_X_train)}, {len(n_X_test)}")
        return(None, None, None)

    #-------------------------
    if model == "AE":

        (AE_model, AE_threshold, AE_recons) = AE.AE_train(X_uncrack=n_X_train, X_crack=n_X_test)

        return(AE_model, AE_threshold, AE_recons)

    #-------------------------
    elif model == "CAE":

        (CAE_model, CAE_threshold, CAE_recons, CAE_train_losses, CAE_healthy_mse, CAE_crack_mse) = CAE.CAE_train(X_uncrack=n_X_train, X_crack=n_X_test)

        return(CAE_model, CAE_threshold, CAE_recons, CAE_train_losses, CAE_healthy_mse, CAE_crack_mse)

    #-------------------------
    else:
        print(f"model type : {model} not recognized") 

    
#============================================================================================================================#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#============================================================================================================================#
if __name__ == '__main__':

    args = tools.arg_parse()
    
    #---------------------------------------------
    if args.train is not None:

        stp.set_config(config_data=stp.get_config(config_path=args.train))
        stp.UTAH_FILES = stp.UTAH_paths(nb_sample=stp.UTAH_FILES_SAMPLE)

        (X_train, X_test, scaler, labels_test) = data.UTAH_data(stp.UTAH_FILES, path_index=3)

        (model, threshold, recons, train_losses, healthy_mse, crack_mse) = run_training(n_X_train=X_train, n_X_test=X_test, model=stp.MODEL)

        #---------------------------------------------
        test_idx = 2

        AE.AE_plot(
            X_crack             = X_test, 
            crack_recon         = recons, 
            warning_threshold   = threshold, 
            index_to_plot       = test_idx
        )
        print(f"Current cracks infos : {labels_test[test_idx]}")

        tools.save_model(model, 
                         scaler,
                         n_threshold=threshold,
                         n_train_losses=train_losses,
                         model_name=stp.MODEL_NAME)

        
        np.savez_compressed(f"./models/{stp.MODEL_NAME}_metrics.npz",
                            train_losses = train_losses,
                            healthy_mse  = healthy_mse,
                            crack_mse    = crack_mse,
                            threshold    = threshold)

    #---------------------------------------------
    if args.test is not None:

        stp.set_config(config_data=stp.get_config(config_path=args.test))

    #---------------------------------------------
    # if args.plot is not None: