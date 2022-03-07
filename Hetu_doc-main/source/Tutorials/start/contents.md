### 快速入门

谢谢你选择Hetu！现在我们以一个CNN模型的训练来走进Hetu。在开始之前，请确保你已经正确地安装了Hetu。

一般地，深度学习任务包含几个重要步骤：1.数据集加载；2.构建模型；3.模型训练；4.模型验证。下面你可以使用Hetu的API，来一步步实现上述步骤。

#### 模型示例

##### **一、导入Hetu**

新建一个python文件，并导入Hetu以及相关的模块。

```python
from hetu import ndarray
from hetu import gpu_ops as ad
from hetu import optimizer
from hetu import dataloader as dl
from hetu import initializers as init
from models.load_data import load_mnist_data, convert_to_one_hot
import numpy as np
```

 

##### **二、加载数据集**

Hetu内置了数据集的加载和切分batch的方法，你可以使用以下方法加载mnist数据集。

```python
executor_ctx = ndarray.gpu(0)

opt = optimizer.SGDOptimizer(learning_rate=0.01)

batch_size = 128

num_epochs = 10

\#data loading

datasets = load_mnist_data("mnist.pkl.gz")

train_set_x, train_set_y = datasets[0]

valid_set_x, valid_set_y = datasets[1]

test_set_x, test_set_y = datasets[2]

 

def local_convert_to_one_hot(batch):

  return convert_to_one_hot(batch, 10)

\# model definition

print('Building model...')

x = dl.dataloader_op([

  dl.Dataloader(train_set_x, batch_size, 'train'),

  dl.Dataloader(valid_set_x, batch_size, 'validate'),

  ])

y_ = dl.dataloader_op([

  dl.Dataloader(train_set_y, batch_size, 'train', local_convert_to_one_hot),

  dl.Dataloader(valid_set_y, batch_size, 'validate', local_convert_to_one_hot),

])
```

​		

##### **三、构建模型**

通过Hetu下的gpu_ops一步步将网络构建起来，Hetu内置了一些常用的初始化工具，你可以方便地进行模型初始化工作。

```python
def conv_relu_avg(x, shape):

  weight = init.random_normal(shape=shape, stddev=0.1)

  x = ad.conv2d_op(x, weight, padding=2, stride=1)

  x = ad.relu_op(x)

  x = ad.avg_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)

  return x

def fc(x, shape):

  weight = init.random_normal(shape=shape, stddev=0.1)

  bias = init.random_normal(shape=shape[-1:], stddev=0.1)

  x = ad.array_reshape_op(x, (-1, shape[0]))

  x = ad.matmul_op(x, weight)

  y = x + ad.broadcastto_op(bias, x)

  return y

 

print('Building CNN model...')

z1 = ad.array_reshape_op(x, [-1, 1, 28, 28])

z2 = conv_relu_avg(z1, (32, 1, 5, 5))

z3 = conv_relu_avg(z2, (64, 32, 5, 5))

y = fc(z3, (7 * 7 * 64, 10))

loss = ad.softmaxcrossentropy_op(y, y_)

loss = ad.reduce_mean_op(loss, [0])

train_op = opt.minimize(loss)
```

##### **四、模型训练**

模型的训练需要定义好Executor，然后直接使用Executor的run方法即可进行训练，并自动按照配置的优化器进行梯度的更新。

```python
executor = ad.Executor([loss, y, train_op], ctx=executor_ctx, dataloader_name='train')

n_train_batches = executor.batch_num

val_executor = ad.Executor([loss, y, y_], ctx=executor_ctx, dataloader_name='validate', inference=True)

n_valid_batches = val_executor.batch_num

\# training

print("Start training loop...")

for i in range(num_epochs):

  print("Epoch %d" % i)

  loss_all = 0

  for minibatch_index in range(n_train_batches):

​    loss_val, predict_y, _ = executor.run()

​    loss_val = loss_val.asnumpy()

​    loss_all += loss_val * x.dataloaders['train'].last_batch_size

  loss_all /= len(train_set_x)

  print("Loss = %f" % loss_all,end='')
```

 

##### **五、模型验证**

你可以使用如下方法，进行模型的验证和推理，并在验证集上评估模型的精度。

```python
#validating

correct_predictions = []

for minibatch_index in range(n_valid_batches):

  loss_val, valid_y_predicted, y_val = val_executor.run(convert_to_numpy_ret_vals=True)

  correct_prediction = np.equal(

​    np.argmax(y_val, 1),

​    np.argmax(valid_y_predicted, 1)).astype(np.float)

  correct_predictions.extend(correct_prediction)

accuracy = np.mean(correct_predictions)

print("\tValidation accuracy = %f" % accuracy)
```

##### **六、结果输出**

对于以上CNN的模型训练和推理，在训练完成之后，你会得到如下的结果，从结果中可以看出，在10个epochs内，模型训练loss不断下降，同时模型在验证集上的精确呈上升趋势。

```
Building model...

Building CNN model...

Start training loop...

Epoch 0

Loss = 0.710760 Validation accuracy = 0.909255

Epoch 1

Loss = 0.319837 Validation accuracy = 0.928786

Epoch 2

Loss = 0.262297 Validation accuracy = 0.939503

Epoch 3

Loss = 0.227060 Validation accuracy = 0.945913

Epoch 4

Loss = 0.201521 Validation accuracy = 0.951623

Epoch 5

Loss = 0.181668 Validation accuracy = 0.956130

Epoch 6

Loss = 0.165777 Validation accuracy = 0.960036

Epoch 7

Loss = 0.152779 Validation accuracy = 0.962139

Epoch 8

Loss = 0.141957 Validation accuracy = 0.964944

Epoch 9

Loss = 0.132787 Validation accuracy = 0.967849
```



#### 模型和ONNX协议间的导入导出

##### **一、 简介**

ONNX (Open Neural Network Exchange) 是一种开源的文件格式，同时也是一种开放式的规范，可以用于存储训练好的模型。使得我们的算法和模型可以在不同的框架之间迁移。

通过ONNX格式，Hetu模型可以转化为ONNX格式，并使用ONNX Runtime等框架进行推理。同时Hetu也支持ONNX转换为Hetu框架模型，并直接用于训练和推理。

模型和ONNX协议之间的导入导出API均放置在hetu.onnx内，你可以进行查询。其中Hetu模型转ONNX协议由**hetu.onnx.hetu2onnx.export**实现，ONNX协议转Hetu模型由**hetu.onnx.onnx2hetu.load_onnx**实现。

下面分别介绍Hetu模型和ONNX之间的导入导出。

##### **二、 模型导出为ONNX协议**

下面以CNN模型为例说明如何将模型转换为ONNX协议。

**2.1模型转出ONNX协议**

模型转换为ONNX协议，首先会根据ONNX的算子协议，将Hetu的算子一一映射为ONNX的算子，而后建图并转换为模型格式保存起来。

```
from hetu import onnx as ax

W1 = init.random_normal((32,1,5,5),stddev=0.1, name='W1')

W2 = init.random_normal((64,32,5,5),stddev=0.1, name='W2')

W3 = init.random_normal((7*7*64,10),stddev=0.1, name='W3')

b3 = init.random_normal((10,),stddev=0.1, name='b3')

X = ad.Variable(name="X")

z1 = ad.conv2d_op(X, W1, padding=2, stride=1)

z2 = ad.relu_op(z1)

z3 = ad.avg_pool2d_op(z2, kernel_H=2, kernel_W=2, padding=0, stride=2)

z4 = ad.conv2d_op(z3, W2, padding=2, stride=1)

z5 = ad.relu_op(z4)

z6 = ad.avg_pool2d_op(z5, kernel_H=2, kernel_W=2, padding=0, stride=2)

z6_flat = ad.array_reshape_op(z6, (-1, 7 * 7 * 64))

y =  ad.matmul_op(z6_flat, W3)+b3

executor=ad.Executor([y],ctx=executor_ctx)

onnx_input_path = 'hetu_cnn_model.onnx'

ax.hetu2onnx.export(executor,[X],[y],onnx_input_path)
```

**2.2 ONNXRuntime推理**

将模型转换为ONNX协议之后，你可以使用ONNXRuntime进行推理，并与Hetu的结果进行比较验证。

```python
import onnxruntime as rt

rand = np.random.RandomState(seed=123)

X_val=rand.normal(scale=0.1, size=(batch_size, 1,28,28)).astype(np.float32)

ath= executor.run(feed_dict={X: X_val})

sess=rt.InferenceSession(onnx_input_path)

input=sess.get_inputs()[0].name

pre=sess.run(None,{input:X_val.astype(np.float32)})[0]

np.testing.assert_allclose(ath[0].asnumpy(),pre,rtol=1e-5)
```

##### **三、** ONNX协议导入Hetu框架

本节以一个Tensorflow的CNN模型转换的ONNX格式为例，说明如何将ONNX协议导入到Hetu框架中。

**3.1 将TF模型保存为ONNX协议**

使用Tensorflow建立模型，利用tf2onnx将其转换为ONNX协议。

```python
import tensorflow.compat.v1 as tf

import tf2onnx

rand = np.random.RandomState(seed=123)

X_val = rand.normal(scale=0.1, size=(20, 784)).astype(np.float32)

with tf.Session() as sess:

  x = tf.placeholder(dtype=tf.float32, shape=(None,784,), name='input')

  z1 = tf.reshape(x, [-1, 28, 28, 1])

  weight1 = tf.Variable(np.random.normal(scale=0.1, size=(32,1,5,5)).transpose([2, 3, 1, 0]).astype(np.float32))

  z2 = tf.nn.conv2d(z1, weight1, padding='SAME', strides=[1, 1, 1, 1])

  z3 = tf.nn.relu(z2)

  z4 = tf.nn.avg_pool(z3, ksize=[1, 2, 2, 1], padding='VALID', strides=[1, 2, 2, 1])

  weight2 = tf.Variable(np.random.normal(scale=0.1, size=(64,32,5,5)).transpose([2, 3, 1, 0]).astype(np.float32))

  z5 = tf.nn.conv2d(z4, weight2, padding='SAME', strides=[1, 1, 1, 1])

  z6 = tf.nn.relu(z5)

  z7 = tf.nn.avg_pool(z6, ksize=[1, 2, 2, 1], padding='VALID', strides=[1, 2, 2, 1])

  z8 = tf.transpose(z7, [0, 3, 1, 2])

  shape = (7 * 7 * 64, 10)

  weight3 = tf.Variable(np.random.normal(scale=0.1, size=shape).astype(np.float32))

  \#bias = tf.Variable(np.random.normal(scale=0.1, size=shape[-1:]).astype(np.float32))

  z9 = tf.reshape(z8, (-1, shape[0]))

  y = tf.matmul(z9, weight3) #+ bias

  _ = tf.identity(y,name='output')

  sess.run(tf.global_variables_initializer())

  expected = sess.run(y, feed_dict={x: X_val})

  graph_def=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def, 'output'])

tf.reset_default_graph()

tf.import_graph_def(graph_def, name='')

model_name='tf_cnn_model.onnx'

with tf.Session() as sess:

  onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=['input:0'], output_names=['output:0'],)

  model_proto = onnx_graph.make_model('cnn_model')

  with open(model_name,'wb') as f:

​    f.write(model_proto.SerializeToString())
```

**3.2 将 ONNX协议转入Hetu框架**

将ONNX协议转入Hetu框架，你可以直接使用hetu.onnx2hetu.load_onnx即可，得到的executor可以用于后续的训练和推理。

```python
x,y =  hx.onnx2hetu.load_onnx(model_name)
executor=ad.Executor([y],ctx=ctx)
```

**3.3 转换结果对比**

使用mnist数据集对上述转换之前的ONNX协议和转换之后的模型进行推理比较，验证转换正确性。其中转换之前的推理使用ONNXRuntime进行。

```python
rand = np.random.RandomState(seed=123)

datasets = load_mnist_data("mnist.pkl.gz")

train_set_x, train_set_y = datasets[0]

valid_set_x, valid_set_y = datasets[1]

test_set_x, test_set_y = datasets[2]

X_val = train_set_x[:20,:]

ath= executor.run(feed_dict={x: X_val})

sess=rt.InferenceSession(model_name)

input=sess.get_inputs()[0].name

pre=sess.run(None,{input:X_val.astype(np.float32)})[0]

np.testing.assert_allclose(ath[0].asnumpy(),pre,rtol=1e-2)

 
```

