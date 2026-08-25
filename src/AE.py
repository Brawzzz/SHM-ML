import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


#================================================================================#
def AE_build(signal_length : int) -> keras.Model:

    """
    Build autoencoder model for signal reconstruction.
    """

    #------------------------------ 
    input_layer = keras.Input(shape=(signal_length,))

    encoded = keras.layers.Dense(128, activation='relu')(input_layer)
    encoded = keras.layers.Dense(64, activation='relu')(encoded)
    encoded = keras.layers.Dense(32, activation='relu')(encoded)

    decoded = keras.layers.Dense(64, activation='relu')(encoded)
    decoded = keras.layers.Dense(128, activation='relu')(decoded)

    output_layer = keras.layers.Dense(signal_length, activation='linear')(decoded)
    
    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

#================================================================================#
def AE_train_validation(X : np.ndarray, y_regression : np.ndarray) -> tuple[keras.Model, float]:

    """
    Auto-Encodeur training and evaluation function. 
    Trains the autoencoder on healthy signals and evaluates its performance on both healthy and cracked signals.
    Visualizes the reconstruction results and highlights areas where the model fails to reconstruct the cracked.

    :param X:                 Input signals (NumPy array).
    :param y_regression:      Labels(NumPy array)

    :return autoencoder:      Trained autoencoder model.
    :return seuil_alerte:     Alert threshold based on reconstruction error.
    """
    
    #------------------------------ 
    scaler      = MinMaxScaler()
    X_scaled    = scaler.fit_transform(X)
    
    #------------------------------ 
    healthy_idx = np.where(y_regression[:, 0] == 0.0)[0]
    crack_idx   = np.where(y_regression[:, 0] > 0.0)[0]
    
    X_uncrack   = X_scaled[healthy_idx]
    X_crack     = X_scaled[crack_idx]
    
    print(f"-> Healthy signals for training : {len(X_uncrack)}")
    print(f"-> Cracks signals for test      : {len(X_crack)}\n")

    (X_train, X_test) = train_test_split(X_uncrack, test_size=0.2, random_state=42)

    #===================================================================#
    #---------------------------- TRAINING -----------------------------#
    #===================================================================#
    signal_size = X.shape[1]
    autoencoder = AE_build(signal_size)
    
    history = autoencoder.fit(x                 = X_train, 
                              y                 = X_train, 
                              epochs            = 100, 
                              batch_size        = 16, 
                              validation_data   = (X_test, X_test),
                              verbose=1)

    #===================================================================#
    #--------------------------- EVALUATION ----------------------------#
    #===================================================================#
    healthy_recon   = autoencoder.predict(X_test)
    healthy_mse     = np.mean(np.square(X_test - healthy_recon), axis=1)
    
    crack_recon     = autoencoder.predict(X_crack)
    crack_mse       = np.mean(np.square(X_crack - crack_recon), axis=1)
    
    print("\RECONSTRUCTION ERROR (MSE) :")
    print(f"-> Health MSE   : {np.mean(healthy_mse):.5f}")
    print(f"-> Crack MSE    : {np.mean(crack_mse):.5f}")
    
    warning_threshold = np.mean(healthy_mse) + 3 * np.std(healthy_mse)
    print(f"\nwarning threshold set at  : {warning_threshold:.5f}")

    #------------------------------
    plt.figure(figsize=(14, 5))
    
    idx_test = 0
    plt.plot(X_crack[idx_test], label='Signal Original (Fissuré)', color='red', alpha=0.7)
    plt.plot(crack_recon[idx_test], label='Tentative de Reconstruction par l\'AE', color='blue', linestyle='--')
    
    error = np.abs(X_crack[idx_test] - crack_recon[idx_test])
    plt.fill_between(range(signal_size), X_crack[idx_test], crack_recon[idx_test], 
                     where=error > (warning_threshold/2), color='orange', alpha=0.3, label='Zone d\'Anomalie Détectée')
    
    plt.title("L'Auto-Encodeur échoue à reconstruire les échos de la fissure")
    plt.xlabel("Temps")
    plt.ylabel("Amplitude (Normalisée)")
    plt.legend()
    plt.show()

    return(autoencoder, warning_threshold)