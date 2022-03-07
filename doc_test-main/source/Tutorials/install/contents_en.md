## Installation Guide

Building Hetu from source code consists the following steps:

### Clone the code 

Users need to clone the code from GitHub:

```
git clone git@github.com:Hsword/Athena.git
```

### Enviroment Requirements

- Python>=3.6
- OpenMP
- CMake>=3.18
- Hetu provides both mkl version and gpu version for users, you could adaptively select from them to meet your needs. The two versions could be installed at the same time. 

**1) MKL version：**

1. MKL 1.6.1

   You could automatically download mkl in Hetu. If you want to manually install mkl, please follow:

   1) downloads: [https://github.com/intel/mkl-dnn/archive/v1.6.1.tar.gz](https://github.com/intel/mkl-dnn/archive/v1.6.1.tar.gz)

   2) compile：

   ```
   mkdir /path/to/build && cd /path/to/build && cmake /path/to/root && make -j8
   ```

2. Change MKL_ROOT into /path/to/root and change MKL_BUILD into /path/to/build in cmake/config.cmake. cmake is the source directory of Hetu. 

**2) GPU version:**

1. CUDA Toolkit 10.1

2. cuDNN v7.5+

3. You could follow these steps to install CUDA and CuDNN:

   1) download CUDA at: [https://developer.nvidia.com](https://developer.nvidia.com)

   2) set Cuda path at cmake/config.cmake

### Installation

According to your installed Hetu version, please change module path in cmake/config.cmake as follows: 

- If you choose MKL version, please add : set(HETU_VERSION "mkl")

- If you choose GPU version, please add : set(HETU_VERSION "gpu")

- If both versions are installed, please add: set(HETU_VERSION "all")

Hetu installation steps are as follows:

​		1) Create a builder file and compile:

```
mkdir build && cd build && cmake ..
```

​		2) If you want to compile all versions of Hetu:

```
make -j 8
```

​		3) If you want to compile Hetu of a specific version, you could specify the version in cmake/config.cmake. And then you can run the command:

```
make hetu -j 8
```

​		4) If you want to comple other versions and modules of Hetu, please follow the summary code below:

```
# generate Makefile
mkdir build && cd build && cmake ..
# compile
# make all
make -j 8
# make hetu, version is specified in cmake/config.cmake
make hetu -j 8
# make allreduce module
make allreduce -j 8
# make ps module
make ps -j 8
# make geometric module
make geometric -j 8
# make hetu-cache module
make hetu_cache -j 8
```

​	5) To delete the compile enviroment, you could use:

```
make clean
```

​	6) Finally, you could activate environment variables for Hetu as follows：

```
source hetu.exp
```