#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits

from scipy.special import i0  # Funzione di Bessel standard e più robusta

# --- Parametri per il Kernel di Gridding ---
# Modifica questi valori a seconda di come sono stati grigliati i dati originali.
# LARGHEZZA_KERNEL è la dimensione in pixel della "finestra" usata per il gridding.
LARGHEZZA_KERNEL = 7 
# ALPHA_KAISER è il parametro di forma del kernel. Un valore comune è 2.3 * LARGHEZZA_KERNEL.
ALPHA_KAISER = 2.3 * LARGHEZZA_KERNEL

def kaiser_bessel_1d(kernel_len, alpha):
    """
    Crea un kernel Kaiser-Bessel 1D in modo standard.
    Usa la funzione i0 (funzione di Bessel modificata di ordine 0) da Scipy.
    """
    if kernel_len % 2 == 0:
        kernel_len += 1 # Assicura che la larghezza sia dispari per avere un centro definito
    
    # Crea un vettore di coordinate da -1 a 1
    x = np.linspace(-1, 1, kernel_len)
    # Calcola l'argomento per la funzione di Bessel
    argomento = alpha * np.sqrt(1 - x**2)
    # Calcola il kernel e lo normalizza al suo picco
    kernel = i0(argomento) / i0(alpha)
    return kernel

def crea_griglia_normalizzazione(dimensione_immagine, larghezza_kernel, alpha):
    """
    Crea la griglia 2D per la normalizzazione calcolando la FFT del kernel.
    """
    # 1. Crea il kernel 1D
    kernel_1d = kaiser_bessel_1d(larghezza_kernel, alpha)
    
    # 2. Crea il kernel 2D usando il prodotto esterno
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # 3. Esegui il "padding": inserisci il piccolo kernel 2D al centro
    #    di una griglia vuota grande quanto l'immagine finale. Questo è un passaggio CRUCIALE.
    griglia_paddata = np.zeros((dimensione_immagine, dimensione_immagine))
    
    centro = dimensione_immagine // 2
    metà_kernel = larghezza_kernel // 2
    
    start = centro - metà_kernel
    end = centro + metà_kernel + 1
    
    griglia_paddata[start:end, start:end] = kernel_2d
    
    # 4. Calcola la FFT del kernel paddato e prendi il valore assoluto
    fft_kernel = np.abs(np.fft.fftshift(np.fft.fft2(griglia_paddata)))
    
    return fft_kernel


filename1 = "output_real.bin"
filename2 = "output_img.bin"
nplanes = 1

with open(filename1, 'rb') as f1:
    vreal  = np.fromfile(f1, dtype=np.float64)
with open(filename2, 'rb') as f2:
    vimg  = np.fromfile(f2, dtype=np.float64)

xaxis = int(np.sqrt(vreal.size))
yaxes = xaxis
residual = np.vectorize(complex)(vreal,vimg)

cumul2d = residual.reshape((xaxis,yaxes,nplanes), order='F')
for i in range(nplanes):
    gridded_orig = np.squeeze(cumul2d[:,:,i])
    gridded_orig = np.rot90(gridded_orig,1)

    # --- Passaggio 1: Crea la griglia di normalizzazione ---
    print("Creazione della griglia di normalizzazione...")
    griglia_norm = crea_griglia_normalizzazione(xaxis, LARGHEZZA_KERNEL, ALPHA_KAISER)

    # --- Passaggio 2: Calcola la FFT dei dati originali ---
    print("Calcolo della FFT...")
    fft_gridded_orig = np.fft.fft2(gridded_orig)
    
    # --- Passaggio 3: Applica la normalizzazione alla FFT ---
    print("Applicazione della normalizzazione...")
    # Esegui la divisione in modo sicuro per evitare errori di divisione per zero
    fft_gridded_normalized = np.divide(fft_gridded_orig, griglia_norm,
                                       where=griglia_norm != 0,
                                       out=np.zeros_like(fft_gridded_orig))
     # --- Passaggio 4: Esegui la Trasformata di Fourier Inversa ---
    print("Calcolo dell'IFT...")
    final_image = np.fft.ifftshift(np.fft.ifft2(fft_gridded_normalized))
    
    ax = plt.subplot()
    #img = ax.imshow(np.abs(np.fft.fftshift(gridded)), aspect='auto', interpolation='none', origin='lower', norm=colors.LogNorm())
    #img = ax.imshow(np.abs(np.fft.fftshift(gridded)), aspect='auto', interpolation='none', origin='lower', vmin=1e0, vmax=3e8)
    img = ax.imshow(np.abs(final_image), aspect='auto', interpolation='none', origin='lower')
    ax.set_xlabel('cell')
    ax.set_ylabel('cell')
    cbar = plt.colorbar(img)
    cbar.set_label('Normalized Image Flux',size=18)
    hdu_fits = fits.PrimaryHDU(np.abs(final_image))
    hdu_fits.writeto('outputs.fits',overwrite=True)
    
    figname='grid_image_real_medium_1' + '.png'
    plt.savefig(figname)
