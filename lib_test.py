USE_MPI = 0

import numpy as np
import casacore.tables as pt
import time
import sys
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
if len(sys.argv) > 1:
    num_threads = int(sys.argv[1])
print("Run with N. threads = ",num_threads)

# set-up the C-python environment
import ctypes
so_gridding = "./gridding.so"
so_fft = "./fft.so"
so_phasecorr = "./phasecorr.so"

c_gridding = ctypes.cdll.LoadLibrary(so_gridding)
#c_fft = ctypes.cdll.LoadLibrary(so_fft)
#c_phasecorr = ctypes.cdll.LoadLibrary(so_phasecorr)
# input MS
readtime0 = time.time()
msfile = "./tail01_10chan.ms"
ms = pt.table(msfile, readonly=True, ack=False)

if rank == 0:
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

#nm_pe = comm.bcast(nm_pe, root=0)
#remaining = comm.bcast(remaining, root=0)

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
grid_size = 64    # number of cells of the grid

# serialize arrays
vis_ser_real = vis.real.flatten()
vis_ser_img = vis.imag.flatten()
uu_ser = uvw[:,0].flatten()
vv_ser = uvw[:,1].flatten()
ww_ser = uvw[:,2].flatten()
weight_ser = weight.flatten()
grid = np.zeros(2*num_w_planes*grid_size*grid_size)  # complex!
gridss = np.zeros(2*num_w_planes*grid_size*grid_size)
gridtot = np.zeros(2*num_w_planes*grid_size*grid_size)  # complex!
image_real = np.zeros(num_w_planes*grid_size*grid_size)
image_imag = np.zeros(num_w_planes*grid_size*grid_size)

#peanokeys = np.empty(vis_ser_real.size,dtype=np.uint64)
gsize = grid.size

hist, bin_edges = np.histogram(ww_ser,num_w_planes)
print(hist)

print(vis_ser_real.dtype)

# normalize uv
minu = np.amin(uu_ser)
maxu = np.amax(uu_ser)
minv = np.amin(vv_ser)
maxv = np.amax(vv_ser)
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

#uu_ser = (uu_ser-ming)/(maxg-ming)
#vv_ser = (vv_ser-ming)/(maxg-ming)
#ww_ser = (ww_ser-minw)/(maxw-minw)
print(uu_ser.shape, vv_ser.dtype, ww_ser.dtype, vis_ser_real.shape, vis_ser_img.dtype, weight_ser.dtype, grid.dtype)

# set normalized uvw - mesh conversion factors
#dx = 1.0/grid_size
#dw = 1.0/num_w_planes

readtime1 = time.time()
# calculate peano keys
t0 = time.time()

# set convolutional kernel parameters full size 
w_support = 7

uu_ser = np.ascontiguousarray(uu_ser)
vv_ser = np.ascontiguousarray(vv_ser)
ww_ser = np.ascontiguousarray(ww_ser)
grid = np.ascontiguousarray(grid)
gridss = np.ascontiguousarray(gridss)
vis_ser_real = np.ascontiguousarray(vis_ser_real)
vis_ser_img = np.ascontiguousarray(vis_ser_img)
weight_ser = np.ascontiguousarray(weight_ser)

# process data


c_gridding.gridding(
              ctypes.c_int(rank),
              ctypes.c_int(size),
              ctypes.c_long(num_points),
              ctypes.c_void_p(uu_ser.ctypes.data),
              ctypes.c_void_p(vv_ser.ctypes.data),
              ctypes.c_void_p(ww_ser.ctypes.data),
              ctypes.c_void_p(grid.ctypes.data),
              ctypes.c_void_p(gridss.ctypes.data),
              ctypes.c_void_p(comm),
              ctypes.c_int(num_threads),
              ctypes.c_int(grid_size),
              ctypes.c_int(grid_size),
              ctypes.c_int(w_support),
              ctypes.c_double(num_w_planes),
              ctypes.c_int(vis.shape[2]),
              ctypes.c_int(vis.shape[1]),
              ctypes.c_void_p(vis_ser_real.ctypes.data),
              ctypes.c_void_p(vis_ser_img.ctypes.data),
              ctypes.c_void_p(weight_ser.ctypes.data),
              ctypes.c_double(minv),
              ctypes.c_double(maxv)
              )


'''
c_fft.fftw_data(
   ctypes.c_int(grid_size),
   ctypes.c_int(grid_size),
   ctypes.c_double(num_w_planes),
   ctypes.c_int(num_threads),
   ctypes.c_void_p(comm),
   ctypes.c_int(size),
   ctypes.c_int(rank),
   ctypes.c_void_p(grid.ctypes.data),
   ctypes.c_void_p(gridss.ctypes.data)
)
'''
c_phasecorr.phase_correction(
   ctypes.c_void_p(gridss.ctypes.data),
   ctypes.c_void_p(image_real.ctypes.data),
   ctypes.c_void_p(image_imag.ctypes.data),
   ctypes.c_double(num_w_planes),
   ctypes.c_int(grid_size),
   ctypes.c_int(grid_size),
   ctypes.c_double(minw),
   ctypes.c_double(maxw),
   ctypes.c_double(minv),
   ctypes.c_double(maxv),
   ctypes.c_int(num_threads),
   ctypes.c_int(size),
   ctypes.c_int(rank)
)


tprocess = time.time() - t0


# reduce results
'''
print("====================================================================")
if USE_MPI == 1:
   nbuffer = 500

   comm.Reduce(grid,gridtot,op=MPI.SUM, root=0)
else:
   gridtot = grid
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
'''
###nnn = 45
###mmm = c_func.test(ctypes.c_int(nnn))
###print("---------> ",nnn,mmm)

#hist, bin_edges = np.histogram(ww_ser,10)
#print(hist)


if rank == 0:
 outfile = "grid"+str(rank)+".txt"
 f = open(outfile, 'w')

# write results
 for iw in range(num_w_planes):
     for iv in range(grid_size):
         for iu in range(grid_size):
            index = 2*(iu + iv*grid_size + iw*grid_size*grid_size)

            v_norm = np.sqrt(gridtot[index]*gridtot[index]+gridtot[index+1]*gridtot[index+1])
            f.writelines(str(iu)+" "+str(iv)+" "+str(iw)+" "+str(gridtot[index])+" "+str(gridtot[index+1])+" "+str(v_norm)+"\n")
 f.close()

#outfile = "data"+str(rank)+".txt"
#f = open(outfile, 'w')
#
#for i in range(nm_pe):
#    #f.writelines(str(uvw_aux[i,0])+" "+str(uvw_aux[i,1])+" "+str(uvw_aux[i,2])+" "+str(peanokeys[i])+"\n")
#    f.writelines(str(uu_ser[i])+" "+str(vv_ser[i])+" "+str(ww_ser[i])+" "+str(peanokeys[i])+"\n")
#
#f.close()

# close MS
ms.close()

# reporting timings
readtime = readtime1-readtime0
if rank == 0:
   print("Read time:       ",readtime)
   print("Process time:    ",tprocess)