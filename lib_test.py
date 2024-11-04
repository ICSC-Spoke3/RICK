USE_MPI = 1

import numpy as np
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
c_fft = ctypes.cdll.LoadLibrary(so_fft)
c_phasecorr = ctypes.cdll.LoadLibrary(so_phasecorr)



with open('newgauss2noconj_t201806301100_SBL180.binMS/ucoord.bin', 'rb') as ufile:
   uu = np.fromfile(ufile, dtype=np.float64)

with open('newgauss2noconj_t201806301100_SBL180.binMS/vcoord.bin', 'rb') as vfile:
   vv = np.fromfile(vfile, dtype=np.float64)

with open('newgauss2noconj_t201806301100_SBL180.binMS/wcoord.bin', 'rb') as wfile:
   ww = np.fromfile(wfile, dtype=np.float64)

with open('newgauss2noconj_t201806301100_SBL180.binMS/visibilities_real.bin', 'rb') as visrealfile:
   vis_real = np.fromfile(visrealfile, dtype=np.float64)

with open('newgauss2noconj_t201806301100_SBL180.binMS/visibilities_img.bin', 'rb') as visimgfile:
   vis_imag = np.fromfile(visimgfile, dtype=np.float64)

with open('newgauss2noconj_t201806301100_SBL180.binMS/weights.bin', 'rb') as weightsfile:
   weight = np.fromfile(weightsfile, dtype=np.float64)

with open('newgauss2noconj_t201806301100_SBL180.binMS/meta.txt', 'r') as metafile:
   for line_num, line in enumerate(metafile):
      if line_num == 0:
         num_points = int(line.strip())
      elif line_num == 1:
         num_vis = int(line.strip())
      elif line_num == 2:
         nchan = int(line.strip())
      elif line_num == 3:
         polarisations = int(line.strip())

metafile.close()

# set convolutional kernel parameters full size 
w_support = 7

# set parameters
num_w_planes = 1 
grid_size = 1024   # number of cells of the grid


# serialize arrays
vis_ser_real = vis_real.flatten()
vis_ser_imag = vis_imag.flatten()
uu_ser = uu.flatten()
vv_ser = vv.flatten()
ww_ser = ww.flatten()
weight_ser = weight.flatten()
grid = np.zeros(2*num_w_planes*grid_size*grid_size, dtype=np.float64)  # complex!
gridss = np.zeros(2*num_w_planes*grid_size*grid_size, dtype=np.float64)
gridtot = np.zeros(2*num_w_planes*grid_size*grid_size)  # complex!
image_real = np.zeros(num_w_planes*grid_size*grid_size, dtype=np.float64)
image_imag = np.zeros(num_w_planes*grid_size*grid_size, dtype=np.float64)


# normalize uv
minu = np.amin(uu_ser)
maxu = np.amax(uu_ser)
minv = np.amin(vv_ser)
maxv = np.amax(vv_ser)
minw = np.amin(ww_ser)
maxw = np.amax(ww_ser)


c_gridding.gridding(
   ctypes.c_int(rank),
   ctypes.c_int(size),
   ctypes.c_int(num_points),
   ctypes.c_void_p(uu_ser.ctypes.data),
   ctypes.c_void_p(vv_ser.ctypes.data),
   ctypes.c_void_p(ww_ser.ctypes.data),
   grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
   gridss.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
   ctypes.py_object(comm),
   ctypes.c_int(num_threads),
   ctypes.c_int(grid_size),
   ctypes.c_int(grid_size),
   ctypes.c_int(w_support),
   ctypes.c_int(num_w_planes),
   ctypes.c_int(polarisations),
   ctypes.c_int(nchan),
   ctypes.c_void_p(vis_ser_real.ctypes.data),
   ctypes.c_void_p(vis_ser_imag.ctypes.data),
   ctypes.c_void_p(weight_ser.ctypes.data),
   ctypes.c_double(minv),
   ctypes.c_double(maxv)
   )

print("Gridding done!")


c_fft.fftw_data(
   ctypes.c_int(grid_size),
   ctypes.c_int(grid_size),
   ctypes.c_double(num_w_planes),
   ctypes.c_int(num_threads),
   ctypes.py_object(comm),
   ctypes.c_int(size),
   ctypes.c_int(rank),
   grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
   gridss.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
)

print("FFT done!")


c_phasecorr.phase_correction(
   grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
   ctypes.c_int(rank),
   ctypes.py_object(comm)
)
print("Phase correction done!")

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

'''
if rank == 0:
 outfile = "grid"+str(rank)+".txt"
 f = open(outfile, 'w')

# write results
 for iw in range(num_w_planes):
     for iv in range(grid_size):
         for iu in range(grid_size):
            index = 2*(iu + iv*grid_size + iw*grid_size*grid_size)

            v_norm = np.sqrt(gridss[index]*gridss[index]+gridss[index+1]*gridss[index+1])
            f.writelines(str(iu)+" "+str(iv)+" "+str(iw)+" "+str(gridss[index])+" "+str(gridss[index+1])+" "+str(v_norm)+"\n")
 f.close()
'''
#outfile = "data"+str(rank)+".txt"
#f = open(outfile, 'w')
#
#for i in range(nm_pe):
#    #f.writelines(str(uvw_aux[i,0])+" "+str(uvw_aux[i,1])+" "+str(uvw_aux[i,2])+" "+str(peanokeys[i])+"\n")
#    f.writelines(str(uu_ser[i])+" "+str(vv_ser[i])+" "+str(ww_ser[i])+" "+str(peanokeys[i])+"\n")
#
#f.close()

# close MS