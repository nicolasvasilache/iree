I'm running the following on macos and avoiding to depend on remote multi-GPU nodes for developing and compiling for now.

# Python env config
```
python -m venv ~/.venv/turbine_dev
source ~/.venv/turbine_dev/bin/activate
```

# IREE config
```
pip install -r ./runtime/bindings/python/iree/runtime/build_requirements.txt -e .
export IREE_SOURCE_DIR=${HOME}/iree
export IREE_BUILD_DIR=${HOME}/iree-build/
alias load_iree='export PATH=${HOME}/iree-build/build/tools/:${PATH}'
alias cmake_configure_iree='(cd ${IREE_SOURCE_DIR} && cmake -G Ninja -B ${IREE_BUILD_DIR} -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_ENABLE_ASSERTIONS=ON -DIREE_ENABLE_SPLIT_DWARF=ON -DIREE_ENABLE_THIN_ARCHIVES=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DIREE_ENABLE_LLD=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DPython3_EXECUTABLE="$(which python3)" -DIREE_BUILD_PYTHON_BINDINGS=ON -DIREE_TARGET_BACKEND_ROCM=ON -DIREE_HAL_DRIVER_HIP=ON -DIREE_TARGET_BACKEND_METAL_SPIRV=OFF -DIREE_ENABLE_ASSERTIONS=ON)'
alias cmake_build_iree='(cd ${IREE_SOURCE_DIR} && cmake --build ${IREE_BUILD_DIR})'
alias cmake_install_iree='(CMAKE_INSTALL_METHOD=ABS_SYMLINK python -m pip install -e ${IREE_BUILD_DIR}/compiler; CMAKE_INSTALL_METHOD=ABS_SYMLINK python -m pip install -e ${IREE_BUILD_DIR}/runtime)'
```

# IREE turbine config
```
# Install editable local projects.
export IREE_TURBINE_SOURCE_DIR=${HOME}/iree-turbine
alias cmake_install_iree_turbine='(cd ${IREE_TURBINE_SOURCE_DIR} && pip install -r requirements.txt -e .)'
```

# Build and install symlinked python files to be editable in dev mode
cmake_build_iree && cmake_install_iree 