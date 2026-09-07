#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import json
import torch
import numpy as np 

from sklearn.model_selection import train_test_split

import AE

import data
import tools 
import setup as stp


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

        training_outputs = AE.AE_train(X_uncrack=X_train, X_crack=X_test)

        (model, train_losses)    = training_outputs[0], training_outputs[3]
        (threshold, recons)      = training_outputs[1], training_outputs[2]
        (healthy_mse, crack_mse) = training_outputs[4], training_outputs[5]

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