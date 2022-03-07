高性能GNN训练
=============================


图神经网络（GNN）是一种功能强大且灵活的神经网络，它利用数据的自然稀疏连接信息。然而，图的稀疏性和不规则性使得在GPU等数据并行硬件上执行有效的GNN计算非常困难。我们发现现实世界中的图通常具有小世界特性，这意味着节点倾向于创建紧密的组，而不是随机地相互连接。这就是为什么我们使用局部感知算法METIS作为图重排序方法来改进局部性。同时，高效地执行GCN可以提高缓存效率和内存性能。 

此外，对于稀疏矩阵乘法（SPMM），河图使用共享内存来加速计算，这意味着同一块中的线程可以共享稀疏矩阵的值。此外，我们还提供了一种混合模式，它利用了图的块稀疏性。通过METIS的重新排序，图形将被划分为块。对于密集块，我们使用密集矩阵进行存储，并使用块稀疏算法进行计算。对于稀疏块，使用CSR格式存储子图，并选择稀疏矩阵乘法来计算乘积。

我们将河图与Deep Graph Library（DGL）和PyTorch Geometric（PyG）进行了比较，同时选择了9个数据集，包括5个小数据集：PubMed、Cora、Citeseer、Coauthor_phy、Blogcatalog；4个大数据集：Reddit、Proteins、Arxiv、Amazon0601。同时使用64个隐藏层节点和4个隐藏层的GCN网络用于训练。结果如下。

|                | Pubmed  | Cora    | Citeseer | Coauthor_phy | Blogcatalog |
|----------------|---------|---------|----------|--------------|-------------|
| DGL            | 0.0118s | 0.0118s | 0.0118s  | 0.0228s      | 0.0128s     |
| PYG            | 0.0053s | 0.0057s | 0.0054s  | 0.0195s      | 0.0078s     |
| Hetu           | <font color='red'>0.0016s</font> | 0.0010s | <font color='red'>0.0011s</font>  | 0.0059s      | <font color='red'>0.0022s</font>     |
| Hetu(reorder)  | <font color='red'>0.0016s</font> | <font color='red'>0.0009s</font> | <font color='red'>0.0011s</font>  | <font color='red'>0.0056s</font>      | 0.0023s     |
| Hetu(hybrid)   | 0.0024s | 0.0011s | 0.0012s  | 0.0062s      | <font color='red'>0.0022s</font>     |


从表中可以看出，对于小图，河图的计算速度比其他两种图形神经网络框架更快，并且仅使用优化的spmm可以获得更好的训练速度，但在混合模式下没有优势。

|                | Reddit  | Proteins | Arxiv   | Amazon0601 |
|----------------|---------|----------|---------|------------|
| DGL            | 0.6135s | 0.2431s  | 0.0392s | 0.0751s    |
| PYG            | oom     | oom      | 0.0605s | 0.1264s    |
| Hetu           | 0.3442s | 0.1850s  | 0.0195s | 0.0302s    |
| Hetu(reorder)	 | 0.1389s | 0.0843s  | <font color='red'>0.0174s</font> | <font color='red'>0.0196s</font>    |
| Hetu(hybrid)	  | <font color='red'>0.1000s</font> | <font color='red'>0.0575s</font>  | 0.0208s | 0.0262s    |

对于稠密的大图，混合模式可以加快训练速度。如果图中的节点是稀疏的，可以通过对图进行重新排序来获得最佳的训练速度。

下面是一个对训练GNN模型的简单Python演示。

### 导入数据集

首先，你需要从文件夹中加载数据集。如果需要，可以对图形重新排序以加快计算速度。
```python
import numpy as np
import scipy.sparse as sp
from graphmix.partition import metis_reorder

dir_name = 'your_dir'
adj = sp.load_npz(dir_name+"adj.npz").tocoo()
features = np.load(dir_name+"features.npy")
labels = np.load(dir_name+"labels.npy")
adj, features, labels = metis_reorder(adj, features, labels)
```

### 设置模型

然后，你可以获得图的参数并定义所需的变量。之后，你可以定义自己的GCN模型。

```python
import hetu as ht
from hetu import init

node_count = adj.shape[0]
num_features = features.shape[1]
num_classes = np.max(labels)+1
hidden_size = 64
  
ctx = ht.gpu(0)
A = ht.Variable(name="A", trainable=False)
AT = ht.Variable(name="AT", trainable=False)
H = ht.Variable(name="H")
W1 = init.xavier_uniform(shape=(num_features, hidden_size), name="W1", trainable=True, ctx=ctx)
W2 = init.xavier_uniform(shape=(hidden_size, num_classes), name="W2", trainable=True, ctx=ctx)
y_ = ht.Variable(name="y_")        
  
z1 = ht.matmul_op(H, W1)  
z2 = ht.spmm_op(A, AT, z1, True) 
z3 = ht.relu_op(z2)       
z4 = ht.matmul_op(z3, W2)       
y = ht.spmm_op(A, AT, z4, True)  
loss = ht.softmaxcrossentropy_op(y, y_)   
opt = ht.optim.AdamOptimizer()
train_op = opt.minimize(loss)
```

### 模型训练

最后，你可以对模型进行相应的训练。

```python
executor = ht.Executor([y,loss,train_op], ctx=ctx)

def convert_to_one_hot(vals, max_val = 0):
    #Helper method to convert label array to one-hot array
    if max_val == 0:
      max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals   

feed_dict = {
  H: ht.array(features, ctx=ctx),
  y_ : ht.array(convert_to_one_hot(labels, max_val=num_classes), ctx=ctx),
  A: ht.sparse_array(adj.data,(adj.row,adj.col),shape=(adj.shape),ctx=ctx),
  AT: ht.sparse_array(adj.data,(adj.col,adj.row),shape=(adj.shape),ctx=ctx),
}
 
epoch_num = 100
for i in range(epoch_num):
    results = executor.run(feed_dict = feed_dict)  
    y_predict = results[0].asnumpy().argmax(axis=1)
    loss = results[1].asnumpy().mean()
    acc = float(np.sum(y_predict==labels)/labels.shape[0])
    print("Epoch :%d , loss:%.4f , acc:%.4f "%(i,loss,acc))
```

输出结果如下：

```python
Epoch :0 , loss:1.9455 , acc:0.1850
Epoch :1 , loss:1.9300 , acc:0.4332
Epoch :2 , loss:1.9120 , acc:0.4417
Epoch :3 , loss:1.8901 , acc:0.4579
Epoch :4 , loss:1.8653 , acc:0.4679
Epoch :5 , loss:1.8379 , acc:0.4742
```

如果你想要使用混合模式，下面是Python示例。

### 导入数据集

为了使用混合模式，你需要分割图形以获得邻接矩阵的密集部分和稀疏部分。此外，还可以设置块大小和密度阈值。
```python
import numpy as np
import scipy.sparse as sp
from graphmix.partition import metis_reorder
from graphmix.partition import split_graph

dir_name = 'your_dir'
adj = sp.load_npz(dir_name+"adj.npz").tocoo()
features = np.load(dir_name+"features.npy")
labels = np.load(dir_name+"labels.npy")
adj, features, labels = metis_reorder(adj, features, labels)
block_size = 32
theta = 0.05
layout, dense_matrix, sparse_coo, features = split_graph(adj, features, block_size = block_size, theta = theta)
```

### 设置模型

矩阵的稀疏部分和稠密部分需要进行单独的计算，再进行相加得到完整结果。

```python
import hetu as ht
from hetu import init

node_count = labels.shape[0]
num_features = features.shape[1]   
num_classes = np.max(labels)+1
hidden_size = 64
   
ctx = ht.gpu(0)
A = ht.Variable(name="A",trainable=False)
AT = ht.Variable(name="AT",trainable=False)
H = ht.Variable(name="H")
W = ht.Variable(name="W")
W1 = init.xavier_uniform(shape=(num_features, hidden_size), name="W1", trainable=True, ctx=ctx)
W2 = init.xavier_uniform(shape=(hidden_size, num_classes), name="W2", trainable=True, ctx=ctx)    
y_ = ht.Variable(name="y_")    
   
z1 = ht.matmul_op(H,W1)  
z2_dense = ht.bsmm_op(z1,W,layout,block_size,True,ctx)
z2_sparse = ht.spmm_op(A, AT, z1) 
z2 = ht.add_op(z2_dense, z2_sparse)
z3 = ht.relu_op(z2)       
z4_dense = ht.bsmm_op(z3,W,layout,block_size,True,ctx)   
z4_sparse = ht.spmm_op(A, AT, z3)
z4 = ht.add_op(z4_dense, z4_sparse)        
y = ht.matmul_op(z4,W2)    
yy = ht.slice_op(y, (0,0), (node_count,num_classes))
loss = ht.softmaxcrossentropy_op(yy, y_)   
opt = ht.optim.AdamOptimizer()
train_op = opt.minimize(loss)
```

### 模型训练

你需要将邻接矩阵的稀疏部分和稠密部分同时送入字典中。

```python
executor = ht.Executor([yy,loss,train_op], ctx=ctx)

def convert_to_one_hot(vals, max_val = 0):
    #Helper method to convert label array to one-hot array
    if max_val == 0:
      max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals   
    
feed_dict = {
    H: ht.array(features, ctx=ctx),
    W : ht.array(dense_matrix, ctx=ctx),
    y_ : ht.array(convert_to_one_hot(labels, max_val=num_classes), ctx=ctx),
    A: ht.sparse_array(sparse_coo.data,(sparse_coo.row,sparse_coo.col),shape=(sparse_coo.shape),ctx=ctx),
    AT: ht.sparse_array(sparse_coo.data,(sparse_coo.col,sparse_coo.row),shape=(sparse_coo.shape),ctx=ctx),
}

epoch_num = 100
for i in range(epoch_num):
    results = executor.run(feed_dict = feed_dict)  
    y_predict = results[0].asnumpy().argmax(axis=1)
    loss = results[1].asnumpy().mean()
    acc = float(np.sum(y_predict==labels)/labels.shape[0])
    print("Epoch :%d , loss:%.4f , acc:%.4f "%(i,loss,acc))
```

输出结果如下：

```python
Epoch :0 , loss:1.9468 , acc:0.0964
Epoch :1 , loss:1.9327 , acc:0.5074
Epoch :2 , loss:1.9174 , acc:0.5214
Epoch :3 , loss:1.8989 , acc:0.5535
Epoch :4 , loss:1.8780 , acc:0.5742
Epoch :5 , loss:1.8551 , acc:0.5916
```

我们的论文正在审查中，我们将尽快公布全部内容。



