- ## HetuML安装手册

  ### 从源码编译安装HetuML

  如果你希望通过编译源码安装HetuML，可以参考HetuML源码仓库的README，并请按照以下步骤进行。

  ### 克隆仓库

  执行以下命令克隆HetuML仓库：

  ```
  git clone https://github.com/ccchengff/HetuML.git
  ```

  ### 环境准备

  - CMake >= 3.11
    - 您可以通过CMak的官网下载地址：https://cmake.org/download/，根据自己的操作系统以及需要版本，选择相应版本进行安装下载。
  - Protobuf >= 3.0.0
    - 您可以通过Protobuf官网地址https://github.com/protocolbuffers/protobuf/releases，根据自己的需要，选择相应版本进行安装下载。
  - OpenMP
    - 您可以通过https://www.open-mpi.org/software/ompi/选择适合的OpenMP版本并下载。

  ### HetuML安装与编译

  在编译之前，请在HetuML的代码目录下执行以下命令：

  ```
  cp cmake/config.cmake.template cmake/config.cmake
  ```

  然后，通过以下命令进行安装与编译：

  ```
  mkdir build && cd build
  cmake ..
  make -j 8
  ```

  最后，通过以下指令设置环境变量：

  ```
  source env.exp
  ```

  ### 使用HetuML

  恭喜您，至此，您已经完成了HetuML的安装工作。

  您可以在快速入门部分查看如何使用HetuML。