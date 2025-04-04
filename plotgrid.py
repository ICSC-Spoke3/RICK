#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
filename1 = "output_real.bin"
filename2 = "output_img.bin"
nplanes = 1

with open(filename1, 'rb') as f1:
    vreal  = np.fromfile(f1, dtype=np.float64)
with open(filename2, 'rb') as f2:
    vimg  = np.fromfile(f2, dtype=np.float64)

xaxis = int(np.sqrt(vreal.size))
yaxes = xaxis
residual = np.vectorize(complex)(vreal, vimg)

cumul2d = residual.reshape((xaxis,yaxes,nplanes), order='F')
for i in range(nplanes):
    gridded = np.squeeze(cumul2d[:,:,i])
    ax = plt.subplot()
    img = ax.imshow(np.abs(np.fft.fftshift(gridded)), aspect='auto', interpolation='none', origin='lower')
    ax.set_xlabel('cell')
    ax.set_ylabel('cell')
    cbar = plt.colorbar(img)
    cbar.set_label('norm(FFT)',size=18)
    figname='grid_image_cpu_medium' + '.png'
    plt.savefig(figname)
