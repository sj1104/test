## 安装

### 从源码编译安装Hetu

如果你希望通过编译源码安装Hetu，可以参考Hetu源码仓库的README，并请按照以下步骤进行。

### 克隆仓库

执行以下命令克隆Hetu仓库:

```
git clone git@github.com:Hsword/Hetu.git
```



### 环境准备

- Python>=3.6

- 你的编译器应支持OpenMP

- CMake>=3.18

- Hetu提供了2个版本供你选择：mkl版本和gpu版本，这2个版本可以同时安装。

**1)mkl版本：**

- MKL 1.6.1

- Hetu仓库提供自动的mkl下载安装指令，如果你选择自行安装mkl，请按照以下步骤进行：

  1) 下载：https://github.com/intel/mkl-dnn/archive/v1.6.1.tar.gz

  2) 编译：

  ```
  mkdir /path/to/build && cd /path/to/build && cmake /path/to/root && make -j8
  ```

  3) 修改MKL_ROOT to /path/to/root and MKL_BUILD to /path/to/build in cmake/config.cmake。其中cmake是Hetu仓库源码目录。

**2)GPU版本:**

- CUDA Toolkit 10.1 配合cuDNN v7.5+

- 请按照以下步骤下载安装CUDA和CUDNN

  1) 下载：https://developer.nvidia.com

  2) 安装CUDA

  3) 在cmake/config.cmake中设置CUDA路径。

### 安装

- 根据你需要安装的Hetu版本，修改cmake/config.cmake中的模块路径。

  1) 安装CPU版本，请修改为：set(HETU_VERSION "mkl")

  2) 安装GPU版本，请修改为：set(HETU_VERSION "gpu")

  3) 两个版本同时安装，请修改为：set(HETU_VERSION "all")

  4) 修改各模块路径。

- Hetu编译安装步骤如下：

  1) 新建build文件夹并编译

```
mkdir build && cd build && cmake ..
```

​		2) 如果编译Hetu所有版本，请执行：

```
make -j 8
```

​		3) 如果编译Hetu特定版本，版本类型在cmake/config.cmake中指定，请执行：

```
make hetu -j 8
```

​		4) 如果编译其它版本和模块，请按照以下汇总代码执行：

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

​		5) 编译环境删除，请执行：

```
make clean
```

- 设置Hetu运行环境变量：

```
source hetu.exp
```

- 至此，恭喜你，你已经完成了Hetu的安装工作。
