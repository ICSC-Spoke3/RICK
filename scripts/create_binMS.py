USE_MPI = 0

import numpy as np
import casacore.tables as pt
import time
import sys
import os

#outpath = '/data/gridding/data/shortgauss_t201806301100_SBH255.binMS/'
print(sys.argv[1])
outpath = "/data/gridding/data/Lofarbig/"+sys.argv[1]+".binMS/"
os.mkdir(outpath)


ufile = 'ucoord.bin'
vfile = 'vcoord.bin'
wfile = 'wcoord.bin'
weights = 'weights.bin'
visrealfile = 'visibilities_real.bin'
visimgfile = 'visibilities_img.bin'
metafile = 'meta.txt'

offset = 0.0

if USE_MPI == 1:
   from mpi4py import MPI
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()
   print(rank,size)
else:
   comm = 0
   rank = 0
   size = 1

num_threads = 1

# input MS
readtime0 = time.time()
#msfile = "/data/Lofar-data/results/L798046_SB244_uv.uncorr_130B27932t_146MHz.pre-cal.ms"
msfile = "/data/Lofar-Luca/results/"+sys.argv[1]+".ms/"
ms = pt.table(msfile, readonly=True, ack=False)

if rank == 0:
 print("Reading ", msfile)
 print("Writing ", outpath)
# load data and metadata
 with pt.table(msfile + '::SPECTRAL_WINDOW', ack=False) as freqtab:
    freq = freqtab.getcol('REF_FREQUENCY')[0] / 1000000.0
    freqpersample = np.mean(freqtab.getcol('RESOLUTION'))
    timepersample = ms.getcell('INTERVAL',0)

 print("Frequencies (MHz)   : ",freq)
 print("Time interval (sec) : ",timepersample)

 with pt.taql("SELECT ANTENNA1,ANTENNA2,sqrt(sumsqr(UVW)),GCOUNT() FROM $ms GROUPBY ANTENNA1,ANTENNA2") as BL:
    ants1, ants2 = BL.getcol('ANTENNA1'), BL.getcol('ANTENNA2')
    Ntime = BL.getcol('Col_4')[0] # number of timesteps
    Nbaselines = len(ants1)

 print("Number of timesteps : ",Ntime)
 print("Total obs time (hrs): ",timepersample*Ntime/3600)
 print("Number of baselines : ",Nbaselines)

#sp = pt.table(msfile+'::LOFAR_ANTENNA_FIELD', readonly=True, ack=False, memorytable=True).getcol('POSITION')

 ant1, ant2 = ms.getcol('ANTENNA1'), ms.getcol('ANTENNA2')

 number_of_measures = Ntime * Nbaselines
 #nm_pe_aux = int(number_of_measures / size)
 #remaining_aux = number_of_measures % size
 nm_pe = np.array(0) 
 nm_pe = int(number_of_measures / size)
 remaining = np.array(0) 
 remaining = number_of_measures % size
 print(nm_pe,remaining)

else:
 nm_pe = None
 remaining = None

if USE_MPI == 1:
 nm_pe = comm.bcast(nm_pe, root=0)
 remaining = comm.bcast(remaining, root=0)

# set the data domain for each MPI rank
startrow = rank*nm_pe

if rank == size-1:
   nm_pe = nm_pe+remaining
print(rank,nm_pe,remaining)

nrow = nm_pe

# read data
uvw = ms.getcol('UVW',startrow,nrow)
vis = ms.getcol('DATA',startrow,nrow)
weight = ms.getcol('WEIGHT_SPECTRUM',startrow,nrow)
print("Freqs per channel   : ",vis.shape[1])
print("Polarizations       : ",vis.shape[2])
print("Number of observations : ",uvw.shape[0])
print("Data size (MB)      : ",uvw.shape[0]*vis.shape[1]*vis.shape[2]*2*4/1024.0/1024.0)

# set parameters
num_points = uvw.shape[0]
num_w_planes = 1 
grid_size = 100    # number of cells of the grid

# serialize arrays
vis_ser_real = vis.real.flatten()
vis_ser_img = vis.imag.flatten()
print("data types: uvw = ",uvw.dtype," vis = ",vis_ser_real.dtype)
#vis_ser = np.zeros(2*vis_ser_real.size)
#for i in range(vis_ser_real.size):
#    vis_ser[2*i]=vis_ser_real[i]
#    vis_ser[2*i+1]=vis_ser_img[i]

uu_ser = uvw[:,0].flatten()
vv_ser = uvw[:,1].flatten()
ww_ser = uvw[:,2].flatten()
weight_ser = weight.flatten()
grid = np.zeros(2*num_w_planes*grid_size*grid_size)  # complex!
gridtot = np.zeros(2*num_w_planes*grid_size*grid_size)  # complex!
peanokeys = np.empty(vis_ser_real.size,dtype=np.uint64)
gsize = grid.size

hist, bin_edges = np.histogram(ww_ser,num_w_planes)
print(hist)

print(vis_ser_real.dtype)

# normalize uv
minu = np.amin(uu_ser)
maxu = np.amax(abs(uu_ser))
minv = np.amin(vv_ser)
maxv = np.amax(abs(vv_ser))
minw = np.amin(ww_ser)
maxw = np.amax(ww_ser)

if USE_MPI == 1:
   maxu_all = np.array(0,dtype=np.float)
   maxv_all = np.array(0,dtype=np.float)
   maxw_all = np.array(0,dtype=np.float)
   minu_all = np.array(0,dtype=np.float)
   minv_all = np.array(0,dtype=np.float)
   minw_all = np.array(0,dtype=np.float)
   comm.Allreduce(maxu, maxu_all, op=MPI.MAX)
   comm.Allreduce(maxv, maxv_all, op=MPI.MAX)
   comm.Allreduce(maxw, maxw_all, op=MPI.MAX)
   comm.Allreduce(minu, minu_all, op=MPI.MIN)
   comm.Allreduce(minv, minv_all, op=MPI.MIN)
   comm.Allreduce(minw, minw_all, op=MPI.MIN)

   ming = min(minu_all,minv_all)
   maxg = max(maxu_all,maxv_all)
   minw = minw_all
   maxw = maxw_all
   ming = ming-offset*ming
   maxg = maxg+offset*maxg
   minw = minw
   maxw = maxw
else:
   ming = min(minu,minv)
   maxg = max(maxu,maxv)
   ming = ming-offset*ming
   maxg = maxg+offset*maxg
   minw = minw
   maxw = maxw

print(maxu,maxv,maxg)
#uu_ser = (uu_ser-ming)/(maxg-ming)
#vv_ser = (vv_ser-ming)/(maxg-ming)
uu_ser = (uu_ser+maxg)/(2*maxg)
vv_ser = (vv_ser+maxg)/(2*maxg)
ww_ser = (ww_ser-minw)/(maxw-minw)
#print(uu_ser.shape, vv_ser.dtype, ww_ser.dtype, vis_ser_real.shape, vis_ser_img.dtype, weight_ser.dtype, grid.dtype)
print(np.amin(uu_ser),np.amax(uu_ser))
print(np.amin(vv_ser),np.amax(vv_ser))
print(np.amin(ww_ser),np.amax(ww_ser))

# set normalized uvw - mesh conversion factors
dx = 1.0/grid_size
dw = 1.0/num_w_planes

readtime1 = time.time()

if rank == 0:
 outfile = outpath+ufile
 uu_ser.tofile(outfile,sep='')
 outfile = outpath+vfile
 vv_ser.tofile(outfile,sep='')
 outfile = outpath+wfile
 ww_ser.tofile(outfile,sep='')
 outfile = outpath+weights
 weight_ser.tofile(outfile,sep='')
 outfile = outpath+weights
 weight_ser.tofile(outfile,sep='')
 outfile = outpath+visrealfile
 vis_ser_real.tofile(outfile,sep='')
 outfile = outpath+visimgfile
 vis_ser_img.tofile(outfile,sep='')
 outfile = outpath+metafile
 f = open(outfile, 'w')
 f.writelines(str(uu_ser.size)+"\n")
 f.writelines(str(vis_ser_real.size)+"\n")
 f.writelines(str(vis.shape[1])+"\n")
 f.writelines(str(vis.shape[2])+"\n")
 f.writelines(str(Ntime)+"\n")
 f.writelines(str(timepersample)+"\n")
 f.writelines(str(timepersample*Ntime/3600)+"\n")
 f.writelines(str(Nbaselines)+"\n")
 f.writelines(str(ming)+"\n")
 f.writelines(str(maxg)+"\n")
 f.writelines(str(minw)+"\n")
 f.writelines(str(maxw)+"\n")
 f.close()


