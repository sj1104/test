High-performance GNN Training
=============================


Graph Neural Networks (GNNs) are powerful and flexible neural networks that use the naturally sparse connectivity information of the data. However, the sparsity and irregularity in graphs make it notoriously difficult to perform efficient GNN computing on data parallel hardware like GPU. We find that real-world graphs usually have the small-world property, which means that nodes tend to create tightly knit groups rather than randomly connect each other. That's why we use the locality-aware algorithm METIS as the graph reorder method to improve the locality. Meanwhile, it can enhance cache-efficiency and memory performance to efficiently execute GCNs. 

In addition, for sparse matrix multiplication(SPMM), Hetu uses shared memory to accelerate the calculation, which means threads in the same block can share the value of sparse matrix. Also, we provide a hybrid mode, which takes advantage of block sparsity of graphs. Through the reorder of METIS, the graph will be partitioned into blocks. For dense blocks, we use dense matrix to store and block-parse algorithm for calculation. For sparse blocks, CSR format is used to store the subgraph and we choose sparse matrix multiplication to calculate the product.

We compare Hetu to Deep Graph Library(DGL)  and PyTorch Geometric (PyG), meanwhile, we selected 9 datasets, including 5 small datasets: PubMed, Cora, Citeseer, Coauthor_phy, Blogcatalog; 4 large datasets: Reddit, Proteins, Arxiv, Amazon0601. The GCN network with 64 hidden layer nodes and 4 hidden layers is used for training. The results are as follows.

|                | Pubmed  | Cora    | Citeseer | Coauthor_phy | Blogcatalog |
|----------------|---------|---------|----------|--------------|-------------|
| DGL            | 0.0118s | 0.0118s | 0.0118s  | 0.0228s      | 0.0128s     |
| PYG            | 0.0053s | 0.0057s | 0.0054s  | 0.0195s      | 0.0078s     |
| Hetu           | <font color='red'>0.0016s</font> | 0.0010s | <font color='red'>0.0011s</font>  | 0.0059s      | <font color='red'>0.0022s</font>     |
| Hetu(reorder)  | <font color='red'>0.0016s</font> | <font color='red'>0.0009s</font> | <font color='red'>0.0011s</font>  | <font color='red'>0.0056s</font>      | 0.0023s     |
| Hetu(hybrid)   | 0.0024s | 0.0011s | 0.0012s  | 0.0062s      | <font color='red'>0.0022s</font>     |


It can be seen from the table that for the small graph, the calculation speed of Hetu is faster than the other two graph neural network frameworks, and only using the optimized spmm can get better training speed, but there is no advantage in the hybrid mode.

|                | Reddit  | Proteins | Arxiv   | Amazon0601 |
|----------------|---------|----------|---------|------------|
| DGL            | 0.6135s | 0.2431s  | 0.0392s | 0.0751s    |
| PYG            | oom     | oom      | 0.0605s | 0.1264s    |
| Hetu           | 0.3442s | 0.1850s  | 0.0195s | 0.0302s    |
| Hetu(reorder)	 | 0.1389s | 0.0843s  | <font color='red'>0.0174s</font> | <font color='red'>0.0196s</font>    |
| Hetu(hybrid)	  | <font color='red'>0.1000s</font> | <font color='red'>0.0575s</font>  | 0.0208s | 0.0262s    |

For dense large graph, the hybrid mode can speed up the training. If the nodes in the graph are sparse, the optimal training speed can be obtained by reorder the graph.

Here is a simple Python demo to show how to train GNN model.

### Load dataset.

First, you need to load dataset from your file folder.If you want, you can reorder the graph to accelerate computing.
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

### Set models.

Then, you can get the parameters of graph and define variables  you need. After that, you can make your own gcn model.

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

### Train models.

Finally, you can set the executor and train your models.

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

The print results are as follows：

```python
Epoch :0 , loss:1.9455 , acc:0.1850
Epoch :1 , loss:1.9300 , acc:0.4332
Epoch :2 , loss:1.9120 , acc:0.4417
Epoch :3 , loss:1.8901 , acc:0.4579
Epoch :4 , loss:1.8653 , acc:0.4679
Epoch :5 , loss:1.8379 , acc:0.4742
```

If you want to use the hybrid mode, here is the simple Python demo.

### Load dataset.

To use the hybird mode, you need to split the graph to get the block-sparse part and coo part of the adjacency matrix. Also, you can set the block size and density threshold.
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

### Set models.

The two parts of the sparse matrix are multiplied separately. After multiplication, add them to get the full result.

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

### Train models.

You need to send both the sparse part of the adjacency matrix and the block-sparse part into the feed dictionary.

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

The print results are as follows：

```python
Epoch :0 , loss:1.9468 , acc:0.0964
Epoch :1 , loss:1.9327 , acc:0.5074
Epoch :2 , loss:1.9174 , acc:0.5214
Epoch :3 , loss:1.8989 , acc:0.5535
Epoch :4 , loss:1.8780 , acc:0.5742
Epoch :5 , loss:1.8551 , acc:0.5916
```

Our paper is under-reviewing and we will release the full details as soon as possible.



