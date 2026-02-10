module purge
module load intel-2021.6.0/udunits/2.2.28-5obkm
module load intel-2021.6.0/libszip/2.1.1-tvhyi
module load intel-2021.6.0/curl/7.85.0-djjip
module load gcc-12.2.0/12.2.0
module load intel-2021.6.0/2021.6.0
module load impi-2021.6.0/2021.6.0
module load intel-2021.6.0/hdf5/1.13.3-xwdun
module load intel-2021.6.0/ncview/2.1.8-sds5t
module load intel-2021.6.0/cmake/3.25.1-7wfsx
module load oneapi-2022.1.0/2022.1.0
module load oneapi-2022.1.0/mkl/2022.1.0
module load intel-2021.6.0/impi-2021.6.0/hdf5-threadsafe/1.13.3-zbgha
module load intel-2021.6.0/impi-2021.6.0/netcdf-fortran-threadsafe/4.6.0-75oow
module load intel-2021.6.0/netcdf-c-threadsafe/4.9.0-25h5k
module load anaconda/3-2022.10
module load intel-2021.6.0/impi-2021.6.0/parmetis/4.0.3-ocsgn

export WWATCH3_NETCDF="NC4"
export NETCDF_CONFIG="/juno/opt/spacks/0.20.0/opt/spack/linux-rhel8-icelake/intel-2021.6.0/intel-oneapi-mpi-2021.6.0/netcdf-c-threadsafe/4.9.0-wpe4ttq72z2yvaw2gjzwzohpe2zu3jik/bin/nc-config"
export METIS_PATH="/juno/opt/spacks/0.20.0/opt/spack/linux-rhel8-icelake/intel-2021.6.0/metis/5.1.0-z5fvkcwoekae4wrpefsszy7abeyhbwlf/"
export PARMETIS_PATH="/juno/opt/spacks/0.20.0/opt/spack/linux-rhel8-icelake/intel-2021.6.0/parmetis/4.0.3-ocsgnm6g4rbw54e4fzycmtlfft2p6v3y"
#source activate work39


for dir in ls ~/.conda/envs/wtest/lib/python3.11/site-packages/nvidia/*; do
        export LD_LIBRARY_PATH="$dir/lib:$LD_LIBRARY_PATH"
done

source activate TF
