#!/bin/bash
#SBATCH -A clusters
#SBATCH -p pleiadi
#SBATCH -J LOFAR
### number of nodes
#SBATCH -N 16
### number of hyperthreading threads
#SBATCH --ntasks-per-core=1
### number of MPI tasks per node
#SBATCH --ntasks-per-node=36
#SBATCH -n 576
### number of openmp threads
#SBATCH --cpus-per-task=1
### number of allocated GPUs per node
#SBATCH --gres=gpu:0
#SBATCH --mem=110G
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -t 08:00:00 


module purge

echo $SLURM_NODELIST
export MODULE_VERSION=5.0.1
source /opt/cluster/spack/share/spack/setup-env.sh
module load default-gcc-11.2.0

cd /u/glacopo/LOFAR/hpc_imaging-end-of-re_structuring/
make SYSTYPE=Amonra -j1 clean
rm -f w-stackingCfftw_ring
make SYSTYPE=Amonra -j1 mpi_ring

export use_cuda=no
  
if [ "$use_cuda" = "no" ]
then
  export typestring=omp_cpu
  export exe=w-stackingCfftw_ring
fi

if [ "$use_cuda" = "yes" ]
then
  export typestring=cuda
  export exe=w-stackingfftw
fi

export logdir=mpi_${SLURM_NTASKS}_${typestring}_${SLURM_CPUS_PER_TASK}
echo "Creating $logdir"
rm -fr $logdir
mkdir $logdir

for itest in {1..2}
do
  export logfile=test_${itest}_${logdir}.log
  echo "time mpirun -np 512 --map-by ppr:32:node /u/glacopo/LOFAR/hpc_imaging-end-of-re_structuring/${exe} /u/glacopo/LOFAR/hpc_imaging-end-of-re_structuring/data/paramfile.txt" > $logfile
  time mpirun -np 512 --map-by ppr:32:node --bind-to core --mca btl self,vader /u/glacopo/LOFAR/hpc_imaging-end-of-re_structuring/${exe} /u/glacopo/LOFAR/hpc_imaging-end-of-re_structuring/data/paramfile.txt >> $logfile
  mv $logfile $logdir
  mv timings.dat ${logdir}/timings_${itest}.dat
  cat ${logdir}/timings_all.dat ${logdir}/timings_${itest}.dat >> ${logdir}/timings_all.dat
done

