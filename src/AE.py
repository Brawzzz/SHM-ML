import numpy as np
import matplotlib.pyplot as plt
import keras

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
def AE_train(X_uncrack: np.ndarray, X_crack: np.ndarray) -> tuple[keras.Model, float]:

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
    print(f"-> Healthy signals for training : {len(X_uncrack)}")
    print(f"-> Cracks signals for test      : {len(X_crack)}\n")

    (X_train, X_val) = train_test_split(X_uncrack, test_size=0.2, random_state=42)

    #===================================================================#
    #---------------------------- TRAINING -----------------------------#
    #===================================================================#
    signal_size = X_uncrack.shape[1]
    autoencoder = AE_build(signal_size)
    
    history = autoencoder.fit(x               = X_train, 
                              y               = X_train, 
                              epochs          = 100, 
                              batch_size      = 16, 
                              validation_data =(X_val, X_val),
                              verbose         = 1
    )

    #===================================================================#
    #--------------------------- EVALUATION ----------------------------#
    #===================================================================#
    healthy_recon = autoencoder.predict(X_val)
    healthy_mse   = np.mean(np.square(X_val - healthy_recon), axis=1)
    
    crack_recon   = autoencoder.predict(X_crack)
    crack_mse     = np.mean(np.square(X_crack - crack_recon), axis=1)
    
    print("\nRECONSTRUCTION ERROR (MSE) :")
    print(f"-> Health MSE   : {np.mean(healthy_mse):.5f}")
    print(f"-> Crack MSE    : {np.mean(crack_mse):.5f}")
    
    warning_threshold = np.mean(healthy_mse) + 3 * np.std(healthy_mse)
    print(f"\nWarning threshold set at  : {warning_threshold:.5f}")

    return(autoencoder, warning_threshold, crack_recon)

#================================================================================#
def AE_plot(X_crack: np.ndarray, crack_recon: np.ndarray, warning_threshold: float, index_to_plot: int = 0):

    """
    Plot a graph comparing the original cracks signal and its reconstruction by the autoencoder. 
    Colour the areas where the reconstruction error exceeds the critical threshold.
    
    :param X_crack:           NumPy array containing des originals crack signals
    :param crack_recon:       NumPy array containing reconstruct signals from AE.
    :param warning_threshold: MSE threshold for an anomaly
    :param index_to_plot:     index of the signal to plot
    """

    #------------------------------ 
    if len(X_crack) == 0 or index_to_plot >= len(X_crack):
        print(f"no signals fouds len(X_crack) = {len(X_crack)} OR index out of range : {index_to_plot}.")
        return

    signal_size = X_crack.shape[1]
    
    plt.figure(figsize=(14, 5))
    
    plt.plot(X_crack[index_to_plot], label='Signal original (Fissuré)', color='red', alpha=0.8, linewidth=1.5)
    plt.plot(crack_recon[index_to_plot], label="Signal reconstruit ", color='blue', linestyle='--', linewidth=1.5)
    
    error_absolue = np.abs(X_crack[index_to_plot] - crack_recon[index_to_plot])
    visual_threshold = warning_threshold / 2 
    
    plt.fill_between(
        range(signal_size), 
        X_crack[index_to_plot], 
        crack_recon[index_to_plot], 
        where=(error_absolue > visual_threshold), 
        color='orange', 
        alpha=0.4, 
        label="Zone d'Anomalie Détectée"
    )
    
    #------------------------------ 
    plt.title(f"Analyse de l'Anomalie - Échec de reconstruction (Index {index_to_plot})", fontsize=14, fontweight='bold')
    plt.xlabel("Échantillons temporels", fontsize=12)
    plt.ylabel("Amplitude (Normalisée)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.show()