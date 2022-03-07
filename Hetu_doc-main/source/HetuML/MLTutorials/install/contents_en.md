- ## HetuML Installation Guide

  Building HetuML from source code consists the following steps:

  ### Clone the code

  Users need to clone the code from GitHub:

  ```
  git clone https://github.com/ccchengff/HetuML.git
  ```

  ### Enviroment Requirements

  - CMake >= 3.11

    downloads:  [https://cmake.org/download/](https://cmake.org/download/)
  - Protobuf >= 3.0.0

    downloads: [https://github.com/protocolbuffers/protobuf/releases](https://github.com/protocolbuffers/protobuf/releases)
  - OpenMP

    downloads: [https://www.open-mpi.org/software/ompi/](https://www.open-mpi.org/software/ompi/)

  ### Build from Source Code

  Users need to execute the following command in the source code folder:

  ```
  cp cmake/config.cmake.template cmake/config.cmake
  ```

  Then, you could compile HetuML:

  ```
  mkdir build && cd build
  cmake ..
  make -j 8
  ```

  If the building is successfull, type this command：

  ```
  source env.exp
  ```

  ### Work with HetuML

  You can work with HetuML now! 

  You could quickly try out HetuML on a small demo in the quick start tutorial. 