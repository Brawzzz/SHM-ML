#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import json
import torch
import numpy as np 

from sklearn.model_selection import train_test_split

import CAE

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

        config = stp.get_config(config_path=args.train)

        stp.set_config(config_data=config)
        stp.print_hyperparameters()

        stp.UTAH_FILES                          = stp.UTAH_paths(nb_sample=stp.UTAH_FILES_SAMPLE)
        (X_train, X_test, scaler, labels_test)  = data.UTAH_data(stp.UTAH_FILES, path_index=3)

        training_outputs = CAE.CAE_train(X_uncrack=X_train, X_crack=X_test)

        (model, train_losses)    = training_outputs[0], training_outputs[3]
        (threshold, recons)      = training_outputs[1], training_outputs[2]
        (healthy_mse, crack_mse) = training_outputs[4], training_outputs[5]

        #---------------------------------------------
        # test_idx = 2

        # CAE.CAE_plot(
        #     X_crack             = X_test, 
        #     crack_recon         = recons, 
        #     warning_threshold   = threshold, 
        #     index_to_plot       = test_idx
        # )
        # print(f"Current cracks infos : {labels_test[test_idx]}")

        tools.save_model(model, scaler, model_name=stp.MODEL_NAME)

        np.savez_compressed(f"./models/{stp.MODEL_NAME}_metrics.npz",
                            train_losses = train_losses,
                            healthy_mse  = healthy_mse,
                            crack_mse    = crack_mse,
                            threshold    = threshold)

    #---------------------------------------------
    elif args.test is not None:

        stp.set_config(config_data=stp.get_config(config_path=args.test))

        # model = tools.load_model(model_path   = "./models/CAE_UTAH_shm.pth",
        #                          model_type   = "PyTorch",
        #                          model_class  = CAE.ConvAutoEncoder,
        #                          scaler_path  = "./models/CAE_UTAH_shm_scaler.pkl",
        #                          model_kwargs = {"signal_length": 2000})

        
        # signal      = X_test[args.test]
        # threshold   = 1.0560758859483052e-05

        # (recons, erreurs, diagnostic) = CAE.CAE_inference(model, X_input=signal, n_threshold=threshold)

    #---------------------------------------------
    elif args.plot is not None:

        metrics = np.load(args.plot)

        tools.model_perf(
            train_losses = metrics["train_losses"],
            healthy_mse  = metrics["healthy_mse"],
            crack_mse    = metrics["crack_mse"],
            threshold    = metrics["threshold"]
        )

    #---------------------------------------------
    # else :
    #   print(f"No command found")