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

