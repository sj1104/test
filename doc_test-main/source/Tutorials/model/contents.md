## 模型库

Hetu提供了各种流行的深度学习模型实现：CNN、Transformer、GNN、Embedded。


### CNN

我们对不同模型使用特定的数据集。
```
MNIST: AlexNet, CNN(3-layer), LeNet, LogisticRegression, LSTM, RNN
CIFAR10: MLP, VGG, ResNet
CIFAR100: VGG, ResNet
```

#### **模型定义**
以AlexNet 为例，模型定义如下：

```python
import hetu as ht
from hetu import init

def conv_bn_relu_pool(x, in_channel, out_channel, name, with_relu=True, with_pool=False):
    #Definition of convolution, batch normalization, relu, max pooling layers.
    weight = init.random_normal(
        shape=(out_channel, in_channel, 3, 3), stddev=0.1, name=name+'_weight')
    bn_scale = init.random_normal(
        shape=(1, out_channel, 1, 1), stddev=0.1, name=name+'_bn_scale')
    bn_bias = init.random_normal(
        shape=(1, out_channel, 1, 1), stddev=0.1, name=name+'_bn_bias')
    x = ht.conv2d_op(x, weight, stride=1, padding=1)
    x = ht.batch_normalization_op(x, bn_scale, bn_bias)
    if with_relu:
        x = ht.relu_op(x)
    if with_pool:
        x = ht.max_pool2d_op(x, kernel_H=2, kernel_W=2, stride=2, padding=[0,0])
    return x

def fc(x, shape, name, with_relu=True):
    #Definition of fully connected layers.
    weight = init.random_normal(shape=shape, stddev=0.1, name=name+'_weight')
    bias = init.random_normal(shape=shape[-1:], stddev=0.1, name=name+'_bias')
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    if with_relu:
        x = ht.relu_op(x)
    return x

def alexnet(x, y_):
    '''
    AlexNet model, for MNIST dataset.

    Parameters:
        x: Variable(hetu.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node), shape (1,)
        y: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print('Building AlexNet model...')
    x = ht.array_reshape_op(x, [-1, 1, 28, 28])
    x = conv_bn_relu_pool(x,   1,  32, 'alexnet_conv1',
                          with_relu=True, with_pool=True)
    x = conv_bn_relu_pool(x,  32,  64, 'alexnet_conv2',
                          with_relu=True, with_pool=True)
    x = conv_bn_relu_pool(x,  64, 128, 'alexnet_conv3',
                          with_relu=True, with_pool=False)
    x = conv_bn_relu_pool(x, 128, 256, 'alexnet_conv4',
                          with_relu=True, with_pool=False)
    x = conv_bn_relu_pool(x, 256, 256, 'alexnet_conv5',
                          with_relu=False, with_pool=True)
    x = ht.array_reshape_op(x, [-1, 256*3*3])
    x = fc(x, (256*3*3, 1024), name='alexnet_fc1', with_relu=True)
    x = fc(x, (1024, 512), name='alexnet_fc2', with_relu=True)
    y = fc(x, (512, 10), name='alexnet_fc3', with_relu=False)
    loss = ht.softmaxcrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    return loss, y
```

#### **模型使用**
以下为运行脚本示例：
```bash
cd examples/cnn/
bash scripts/hetu_1gpu.sh mlp CIFAR10   # mlp with CIFAR10 dataset in hetu
bash scripts/hetu_8gpu.sh mlp CIFAR10   # mlp with CIFAR10 in hetu with 8-GPU (1-node)
bash scripts/hetu_16gpu.sh mlp CIFAR10  # mlp with CIFAR10 in hetu with 8-GPU (2-nodes)            
```
如果需要在PS模式下进行训练，你需要首先启动调度器和服务器。

你可以在脚本中更改设置。见“mnist_mlp.sh”。
```bash
#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py

### validate and timing
python ${mainpy} --model mlp --dataset CIFAR10  --validate --timing

### run in cpu
# python ${mainpy} --model mlp --dataset CIFAR10 --gpu -1 --validate --timing

```




### Transformer

我们提供Transformer模型，具体实现如下：

#### **模型定义**


```python
import hetu as ht
from hetu import init
import numpy as np


def layer_norm(input_tensor, feature_size, eps=1e-8):
    #Definition of layer normalization .
    scale = init.ones(name='layer_norm_scale', shape=(feature_size, ))
    bias = init.zeros(name='layer_norm_biad', shape=(feature_size, ))
    return ht.layer_normalization_op(input_tensor, scale, bias, eps=eps)


def dense(input_tensor, fan_in, fan_out, activation=None, kernel_initializer=init.xavier_normal, bias_initializer=init.zeros):
    #Definition of dense layers.
    weights = kernel_initializer(name='dense_weights', shape=(fan_in, fan_out))
    bias = bias_initializer(name='dense_bias', shape=(fan_out,))
    outputs = ht.matmul_op(input_tensor, weights)
    outputs = outputs + ht.broadcastto_op(bias, outputs)
    if activation is not None:
        outputs = activation(outputs)
    return outputs


def dropout(input_tensor, dropout_prob):
    #Definition of dropout layers.
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = ht.dropout_op(input_tensor, 1.0 - dropout_prob)
    return output


def get_token_embeddings(vocab_size, num_units, initializer=init.xavier_normal, zero_pad=True):
    #token embeddings initialize.
    if zero_pad:
        embedding_part = initializer(
            name='embedding_table', shape=(vocab_size-1, num_units))
        padding_zero = init.zeros(
            name='padding_zero', shape=(1, num_units), trainable=False)
        embeddings = ht.concat_op(padding_zero, embedding_part)
    else:
        embeddings = initializer(
            name='embedding_table', shape=(vocab_size, num_units))
    return embeddings


def multihead_attention(queries, keys, values, config, query_act=None, key_act=None, value_act=None, attention_mask=None, causality=False):
    #Definition of attention layers.
    def transpose_for_scores(input_tensor):
        output_tensor = ht.array_reshape_op(
            input_tensor, [config.batch_size, -1, config.num_heads, config.d_model // config.num_heads])

        output_tensor = ht.transpose_op(output_tensor, [0, 2, 1, 3])
        return output_tensor

    batch_size = config.batch_size
    hidden_size = config.d_model
    num_attention_heads = config.num_heads
    caus_len = config.maxlen2 - 1
    attention_probs_dropout_prob = config.dropout_rate

    size_per_head = hidden_size // num_attention_heads

    # reshape to 2d
    queries2d = ht.array_reshape_op(
        queries, [-1, hidden_size])  # (N * T_q, d_model)
    keys2d = ht.array_reshape_op(keys, [-1, hidden_size])  # (N * T_k, d_model)
    values2d = ht.array_reshape_op(
        values, [-1, hidden_size])  # (N * T_k, d_model)

    # linear transformation
    query_layer = dense(queries2d, hidden_size, hidden_size,
                        query_act)  # (N * T_k, d_model)
    key_layer = dense(keys2d, hidden_size, hidden_size,
                      key_act)  # (N * T_k, d_model)
    value_layer = dense(values2d, hidden_size, hidden_size,
                        value_act)  # (N * T_k, d_model)

    # transpose
    query_layer = transpose_for_scores(query_layer)  # (N, h, T_q, d_model/h)
    key_layer = transpose_for_scores(key_layer)  # (N, h, T_k, d_model/h)
    value_layer = transpose_for_scores(value_layer)  # (N, h, T_k, d_model/h)

    # score
    attention_scores = ht.batch_matmul_op(
        query_layer, key_layer, trans_B=True)  # (N, h, T_q, T_k)
    attention_scores = attention_scores * (1.0 / np.sqrt(float(size_per_head)))

    # mask
    if attention_mask is not None:
        zeros = ht.Variable('no_mask', value=np.array(
            (0,), dtype=np.float32), trainable=False)
        adder = ht.Variable('attention_mask', value=np.array(
            (-2**32+1,), dtype=np.float32), trainable=False)
        zeros = ht.broadcastto_op(zeros, attention_mask)
        adder = ht.broadcastto_op(adder, attention_mask)
        attention_mask = ht.where_op(attention_mask, zeros, adder)  # (N, T)
        attention_mask = ht.array_reshape_op(
            attention_mask, [batch_size, 1, 1, -1])
        attention_scores = attention_scores + \
            ht.broadcastto_op(attention_mask, attention_scores)
    if causality:
        tril = ht.Variable(name='tril', value=np.tril(
            np.ones((caus_len, caus_len))), trainable=False)  # (T, T)
        future_masks = ht.broadcast_shape_op(
            tril, [batch_size, num_attention_heads, caus_len, caus_len])
        adder = ht.Variable('future_mask', value=np.array(
            (-2**32+1,), dtype=np.float32), trainable=False)
        adder = ht.broadcastto_op(adder, future_masks)
        attention_scores = ht.where_op(
            future_masks, attention_scores, adder)  # (N, h, T, T)

    # probs
    attention_probs = ht.softmax_op(attention_scores)
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
    context_layer = ht.batch_matmul_op(attention_probs, value_layer)
    context_layer = ht.transpose_op(context_layer, [0, 2, 1, 3])
    outputs = ht.array_reshape_op(
        context_layer,
        [batch_size, -1, num_attention_heads * size_per_head])

    # Residual connection
    outputs = outputs + queries  # (N, T_q, d_model)

    # Normalize
    outputs = layer_norm(outputs, hidden_size)  # (N, T_q, d_model)
    return outputs


def ff(inputs, config):
    outputs = ht.array_reshape_op(inputs, [-1, config.d_model])
    outputs = dense(outputs, config.d_model,
                    config.d_ff, activation=ht.relu_op)
    outputs = dense(outputs, config.d_ff, config.d_model)
    outputs = ht.array_reshape_op(
        outputs, [config.batch_size, -1, config.d_model])
    outputs = outputs + inputs
    outputs = layer_norm(outputs, config.d_model)
    return outputs


def label_smoothing(inputs, V, epsilon=0.1):
    # V = inputs.shape[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)


def positional_encoding(inputs, inputs_shape, maxlen, masking=True):
    N, T, E = tuple(inputs_shape)
    position_enc = np.array([
        [pos / np.power(10000, (i & -2)/E) for i in range(E)]
        for pos in range(maxlen)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

    position_enc = position_enc[:T, :]
    outputs = ht.Variable(name='position_enc', value=np.tile(
        position_enc, [N, 1, 1]), trainable=False)
    zeros = ht.Variable(name='zeros', value=np.zeros(
        inputs_shape), trainable=False)

    if masking:
        outputs = ht.where_op(inputs, outputs, zeros)

    return outputs


class Transformer(object):
    '''
    Transformer model.

    Parameters:
        xs: Variable(hetu.gpu_ops.Node.Node)
        ys: Variable(hetu.gpu_ops.Node.Node)
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node)
    '''
    def __init__(self, hp):
        self.hp = hp
        self.embeddings = get_token_embeddings(
            self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def encode(self, xs):
        x = xs

        # embedding
        enc = ht.embedding_lookup_op(self.embeddings, x)  # (N, T1, d_model)
        enc = enc * self.hp.d_model**0.5  # scale

        enc += positional_encoding(enc, (self.hp.batch_size,
                                         self.hp.maxlen1, self.hp.d_model), self.hp.maxlen1)
        enc = dropout(enc, self.hp.dropout_rate)

        # Blocks
        for i in range(self.hp.num_blocks):
            # self-attention
            enc = multihead_attention(
                queries=enc, keys=enc, values=enc,
                config=self.hp,
                attention_mask=x,
                causality=False
            )
            # feed forward
            enc = ff(enc, config=self.hp)
        memory = enc
        return memory

    def decode(self, ys, memory, src_masks):
        decoder_inputs = ys

        # embedding
        dec = ht.embedding_lookup_op(
            self.embeddings, decoder_inputs)  # (N, T2, d_model)
        dec = dec * self.hp.d_model ** 0.5  # scale

        dec += positional_encoding(dec, (self.hp.batch_size,
                                         self.hp.maxlen2-1, self.hp.d_model), self.hp.maxlen2)
        dec = dropout(dec, self.hp.dropout_rate)

        # Blocks
        for i in range(self.hp.num_blocks):
            # Masked self-attention (Note that causality is True at this time)
            dec = multihead_attention(
                queries=dec, keys=dec, values=dec,
                config=self.hp,
                attention_mask=decoder_inputs,
                causality=True,
            )
            # Vanilla attention
            dec = multihead_attention(
                queries=dec, keys=memory, values=memory,
                config=self.hp,
                attention_mask=src_masks,
                causality=False,
            )
            # Feed Forward
            dec = ff(dec, config=self.hp)

        dec = ht.array_reshape_op(
            dec, [-1, self.hp.d_model])  # (N * T, d_model)
        logits = ht.array_reshape_op(ht.matmul_op(dec, self.embeddings, trans_B=True), [
                                     self.hp.batch_size, -1, self.hp.vocab_size])  # (N, T, vocab)

        return logits

    def train(self, xs, ys):
        # forward
        memory = self.encode(xs)
        logits = self.decode(ys[0], memory, xs)

        # train scheme
        y = ys[1]
        y_ = label_smoothing(ht.one_hot_op(
            y, self.hp.vocab_size), self.hp.vocab_size)  # (N, T, vocab)
        loss = ht.softmaxcrossentropy_op(logits, y_)

        return loss

```

#### **模型使用**

你可以使用以下命令运行Transformer：
```bash
cd examples/nlp/
python train_hetu_transformer.py
```
如果你需要更改超参数，请修改`hparams.py` 文件。


### GNN
我们提供了一个简单的2层GCN模型，具体实现如下：

#### **模型定义**

```python
import hetu as ht
from hetu import init

class GCN(object):
    #Definition of GCN layers.
    def __init__(self, in_features, out_features, norm_adj, activation=None, dropout=0,
                 name="GCN", custom_init=None):
        if custom_init is not None:
            self.weight = ht.Variable(
                value=custom_init[0], name=name+"_Weight")
            self.bias = ht.Variable(value=custom_init[1], name=name+"_Bias")
        else:
            self.weight = init.xavier_uniform(
                shape=(in_features, out_features), name=name+"_Weight")
            self.bias = init.zeros(shape=(out_features,), name=name+"_Bias")
        # self.mp is a sparse matrix and should appear in feed_dict later
        self.mp = norm_adj
        self.activation = activation
        self.dropout = dropout
        self.output_width = out_features

    def __call__(self, x):
        """
            Build the computation graph, return the output node
        """
        if self.dropout > 0:
            x = ht.dropout_op(x, 1 - self.dropout)
        x = ht.matmul_op(x, self.weight)
        msg = x + ht.broadcastto_op(self.bias, x)
        x = ht.csrmm_op(self.mp, msg)
        if self.activation == "relu":
            x = ht.relu_op(x)
        elif self.activation is not None:
            raise NotImplementedError
        return x
 
 def convert_to_one_hot(vals, max_val=0):
    #Helper method to convert label array to one-hot array.
    if max_val == 0:
        max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals


def sparse_model(int_feature, hidden_layer_size, embedding_idx_max, embedding_width, num_classes, lr):
    #Definition of 2-layer GCN model.
    y_ = ht.GNNDataLoaderOp(lambda g: ht.array(convert_to_one_hot(
        g.i_feat[:, -2], max_val=num_classes), ctx=ht.cpu()))
    mask_ = ht.Variable(name="mask_")
    index_ = ht.GNNDataLoaderOp(lambda g: ht.array(
        g.i_feat[:, 0:-2], ctx=ht.cpu()), ctx=ht.cpu())
    embedding = ht.init.random_normal(
        [embedding_idx_max, embedding_width], stddev=0.1)
    embed = ht.embedding_lookup_op(embedding, index_)
    feat = ht.array_reshape_op(embed, (-1, int_feature * embedding_width))

    norm_adj_ = ht.Variable("message_passing", trainable=False, value=None)
    gcn1 = GCN(int_feature * embedding_width,
               hidden_layer_size, norm_adj_, activation="relu")
    gcn2 = GCN(gcn1.output_width, num_classes, norm_adj_)
    x = gcn1(feat)
    y = gcn2(x)
    loss = ht.softmaxcrossentropy_op(y, y_)
    train_loss = loss * mask_
    train_loss = ht.reduce_mean_op(train_loss, [0])
    opt = ht.optim.SGDOptimizer(lr)
    train_op = opt.minimize(train_loss)
    # model input & model output
    return [loss, y, train_op], [mask_, norm_adj_]

```

#### **模型使用**
你可以使用以下命令运行GNN模型：
```bash
cd /examples/gnn/
python run_single.py -p ~/yourDataPath/Reddit [--sparse]                                                     # run locally
python run_dist.py [configfile] -p ~/yourDataPath/Reddit [--sparse]                                          # run in ps setting (locally)
python run_dist_hybrid.py [configfile] -p ~/yourDataPath/Reddit --server                                     # run in hybrid setting (multi device)
mpirun -np 4 --allow-run-as-root python3 run_dist_hybrid.py [configfile] -p ~/yourDataPath/Reddit [--sparse] # run in hybrid setting (locally)
```

### Embedding
我们提供了两种推荐模型：CTR和NCF。

#### **模型定义**
以NCF为例，具体模型定义如下：
```python
import hetu as ht
from hetu import init
import numpy as np

def neural_mf(user_input, item_input, y_, num_users, num_items):
    #Definition of NCF model.
    embed_dim = 8
    layers = [64, 32, 16, 8]
    learning_rate = 0.01

    User_Embedding = init.random_normal(
        (num_users, embed_dim + layers[0] // 2), stddev=0.01, name="user_embed", ctx=ht.cpu(0))
    Item_Embedding = init.random_normal(
        (num_items, embed_dim + layers[0] // 2), stddev=0.01, name="item_embed", ctx=ht.cpu(0))

    user_latent = ht.embedding_lookup_op(
        User_Embedding, user_input, ctx=ht.cpu(0))
    item_latent = ht.embedding_lookup_op(
        Item_Embedding, item_input, ctx=ht.cpu(0))

    mf_user_latent = ht.slice_op(user_latent, (0, 0), (-1, embed_dim))
    mlp_user_latent = ht.slice_op(user_latent, (0, embed_dim), (-1, -1))
    mf_item_latent = ht.slice_op(item_latent, (0, 0), (-1, embed_dim))
    mlp_item_latent = ht.slice_op(item_latent, (0, embed_dim), (-1, -1))

    W1 = init.random_normal((layers[0], layers[1]), stddev=0.1, name='W1')
    W2 = init.random_normal((layers[1], layers[2]), stddev=0.1, name='W2')
    W3 = init.random_normal((layers[2], layers[3]), stddev=0.1, name='W3')
    W4 = init.random_normal((embed_dim + layers[3], 1), stddev=0.1, name='W4')

    mf_vector = ht.mul_op(mf_user_latent, mf_item_latent)
    mlp_vector = ht.concat_op(mlp_user_latent, mlp_item_latent, axis=1)
    fc1 = ht.matmul_op(mlp_vector, W1)
    relu1 = ht.relu_op(fc1)
    fc2 = ht.matmul_op(relu1, W2)
    relu2 = ht.relu_op(fc2)
    fc3 = ht.matmul_op(relu2, W3)
    relu3 = ht.relu_op(fc3)
    concat_vector = ht.concat_op(mf_vector, relu3, axis=1)
    y = ht.matmul_op(concat_vector, W4)
    y = ht.sigmoid_op(y)
    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)
    return loss, y, train_op

```

#### **模型使用**
你可以使用以下命令运行嵌入模型：
```bash
cd examples/rec/
python run_hetu.py    # run locally
bash ps_ncf.sh        # run in ps setting (locally)
bash hybrid_ncf.sh    # run in hybrid setting (locally)
```

我们将添加更多的模型示例。

