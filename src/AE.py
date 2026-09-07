#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
from xml.parsers.expat import model

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import setup as stp


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class Autoencoder(nn.Module):

    """
    This class defines the architecture of an auto-encodeur based torch.nn.Module
    """

    def __init__(self, signal_length : int):

        super(Autoencoder, self).__init__()
        
        #------------------------------
        self.encoder = nn.Sequential(

            nn.Linear(signal_length, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        #------------------------------
        self.decoder = nn.Sequential(

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),
            
            nn.Linear(128, signal_length)
        )

    #================================================================================#
    def forward(self, x):

        """
        pass the signal in the auto-encodeur network

        Parameters
        ----------
        x : the input signal 
        """

        #---------------------------------------------
        x = self.encoder(x)
        x = self.decoder(x)

        return x

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def AE_train(X_uncrack: np.ndarray, X_crack: np.ndarray) -> tuple[nn.Module, float, np.ndarray, np.ndarray]:

    """
    Auto-Encodeur training and evaluation function. 
    Trains the autoencoder on healthy signals and evaluates it's performances on both healthy and cracked signals.

    Parameters
    ----------
    X_uncrack : Input signals (NumPy array)
    X_crack   : Labels(NumPy array)

    Returns
    ----------
    model               : Trained autoencoder model.
    warning_threshold   : Alert threshold based on reconstruction error.
    crack_recon         : Reconstructed crack signals.
    train_losses        : Training losses. 
    """

    #--------------------------------------------- 
    print(f"-> Healthy signals for training : {len(X_uncrack)}")
    print(f"-> Cracks signals for test      : {len(X_crack)}\n")

    (X_train, X_val) = train_test_split(X_uncrack, test_size=0.2, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Training set on : {device}")

    #------------------------------
    tensor_train = torch.tensor(X_train, dtype=torch.float32)
    tensor_val   = torch.tensor(X_val, dtype=torch.float32)
    tensor_crack = torch.tensor(X_crack, dtype=torch.float32)

    #------------------------------
    dataset     = TensorDataset(tensor_train, tensor_train)
    dataloader  = DataLoader(dataset, batch_size=stp.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    #------------------------------
    signal_size = X_uncrack.shape[1]
    model       = Autoencoder(signal_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=stp.LEARNING_RATE)

    #===================================================================#
    #---------------------------- TRAINING -----------------------------#
    #===================================================================#
    pbar = tqdm(range(stp.EPOCHS), desc="Training AE", unit="epoch")

    model.train()
    train_losses = []

    #------------------------------
    for epoch in range(stp.EPOCHS):

        train_loss = 0.0

        #---------------
        for batch_x, batch_y in dataloader:

            optimizer.zero_grad()        

            outputs = model(batch_x)
            loss    = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(dataloader.dataset)
        train_losses.append(train_loss)

        pbar.set_postfix({'Loss MSE': f"{train_loss:.5f}"})
        pbar.update(1)

    #===================================================================#
    #--------------------------- EVALUATION ----------------------------#
    #===================================================================#
    model.eval()
    with torch.no_grad():
        healthy_recon_tensor = model(tensor_val)
        crack_recon_tensor   = model(tensor_crack)
        
    #------------------------------
    healthy_recon = healthy_recon_tensor.numpy()
    crack_recon   = crack_recon_tensor.numpy()
    
    #------------------------------
    healthy_mse = np.mean(np.square(X_val - healthy_recon), axis=1)
    crack_mse   = np.mean(np.square(X_crack - crack_recon), axis=1)
    
    print("\nRECONSTRUCTION ERROR AE (MSE) :")
    print(f"-> Health MSE   : {np.mean(healthy_mse):.5f}")
    print(f"-> Crack MSE    : {np.mean(crack_mse):.5f}")
    
    warning_threshold = np.mean(healthy_mse) + 3 * np.std(healthy_mse)
    print(f"\nWarning threshold set at  : {warning_threshold:.5f}")

    return(model, warning_threshold, crack_recon, train_losses, healthy_mse, crack_mse)

#================================================================================#
def AE_plot(X_crack: np.ndarray, 
            crack_recon: np.ndarray, 
            warning_threshold: float, 
            index_to_plot: int = 0) -> None:

    """
    Plot a graph comparing the original cracks signal and its reconstruction by the autoencoder. 
    Colour the areas where the reconstruction error exceeds the critical threshold.
    
    Parameters
    ----------
    X_crack:           NumPy array containing des originals crack signals
    crack_recon:       NumPy array containing reconstruct signals from AE.
    warning_threshold: MSE threshold for an anomaly
    index_to_plot:     index of the signal to plot
    """

    #--------------------------------------------- 
    if len(X_crack) == 0 or index_to_plot >= len(X_crack):
        print(f"no signals fouds len(X_crack) = {len(X_crack)} OR index out of range : {index_to_plot}.")
        return

    signal_size = X_crack.shape[1]
    
    plt.figure(figsize=(14, 5))
    
    plt.plot(X_crack[index_to_plot], label='Signal original (Fissuré)', color='red', alpha=0.8, linewidth=1.5)
    plt.plot(crack_recon[index_to_plot], label="Signal reconstruit ", color='blue', linestyle='--', linewidth=1.5)
    
    error_abs           = np.abs(X_crack[index_to_plot] - crack_recon[index_to_plot])
    visual_threshold    = 2.5 * np.sqrt(warning_threshold)
    
    plt.fill_between(
        range(signal_size), 
        -2.0,
        2.5,
        where=(error_abs > visual_threshold), 
        color='orange', 
        alpha=0.3,
        zorder=0,
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