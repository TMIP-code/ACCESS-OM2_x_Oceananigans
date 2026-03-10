#!/usr/bin/env bash
set -euo pipefail

# One-time MPItrampoline setup script. Run on the login node.
# Builds MPIwrapper (bridges MPItrampoline_jll to system OpenMPI 5.0.8),
# then configures MPIPreferences to use MPItrampoline_jll.
#
# After running this, precompile packages:
#   bash scripts/pkg_update_project.sh

repo_root=/home/561/bp3051/Projects/TMIP/ACCESS-OM2_x_Oceananigans
cd "$repo_root"

# 1. Load modules needed to build MPIwrapper
module load openmpi/5.0.8
module load cmake/3.24.2

# 2. Build MPIwrapper against system OpenMPI
MPIWRAPPER_INSTALL=$HOME/mpiwrapper
echo "Building MPIwrapper into $MPIWRAPPER_INSTALL"

if [ -d MPIwrapper ]; then
    echo "MPIwrapper source directory already exists, pulling latest"
    cd MPIwrapper && git fetch && git checkout v2.11.0 && cd ..
else
    git clone https://github.com/eschnett/MPIwrapper
    cd MPIwrapper && git checkout v2.11.0 && cd ..
fi

cmake -S MPIwrapper -B MPIwrapper/build \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_Fortran_COMPILER=mpifort \
    -DMPIEXEC_EXECUTABLE=mpiexec \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="$MPIWRAPPER_INSTALL"
cmake --build MPIwrapper/build
cmake --install MPIwrapper/build

echo "MPIwrapper installed to $MPIWRAPPER_INSTALL"
echo "Checking linkage:"
ldd "$MPIWRAPPER_INSTALL/lib64/libmpiwrapper.so" | grep -i mpi

# 3. Configure MPIPreferences to use MPItrampoline_jll
echo "Configuring MPIPreferences to use MPItrampoline_jll"
julia --project -e 'using MPIPreferences; MPIPreferences.use_jll_binary("MPItrampoline_jll")'

# 4. Clean up build directory
rm -rf MPIwrapper

echo "Done! MPItrampoline setup complete."
echo "MPITRAMPOLINE_LIB=$MPIWRAPPER_INSTALL/lib64/libmpiwrapper.so"
echo "This is set automatically by env_defaults.sh for all jobs."
