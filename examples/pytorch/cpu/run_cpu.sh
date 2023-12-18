export OMP_NUM_THREADS=32
export CCL_ATL_TRANSPORT=mpi

mpiexec.hydra -np 4 -ppn 4 python cpu_ft.py