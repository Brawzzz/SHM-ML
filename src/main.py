#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import AE
import data
import tools 
import setup as stp


#============================================================================================================================#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#============================================================================================================================#
cracks_files = ["./data/cracks/L10.xlsx", "./data/cracks/L20.xlsx", 
                "./data/cracks/L30.xlsx", "./data/cracks/L50.xlsx"]

(X_train, scaler)           = data.baseline_data(baseline_file=stp.DATASET_PATH)
(X_test, labels_anomalies)  = data.cracks_data(cracks_files, scaler=scaler, ref_size=668)

#---------------------------------------------
if X_train is None or X_test is None:
    raise ValueError("X_train or X_test is None : X_train : {X_train}, X_test : {X_test}")

(AE_model, threshold, signals_recon) = AE.AE_train(X_uncrack=X_train, X_crack=X_test)

tools.save_model(AE_model, scaler, model_name="AE_model")
AE.AE_plot(X_crack=X_test, crack_recon=signals_recon, warning_threshold=threshold, index_to_plot=0)