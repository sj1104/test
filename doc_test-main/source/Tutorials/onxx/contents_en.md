## Converting to ONNX format

Open Neural Network Exchange (ONNX) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. ONNX provides an open source format for AI models.This allows us to migrate our algorithms and models between different frameworks.

Hetu models can convert to ONNX format, and inference using ONNX Runtime. 

ONNX format can convert to Hetu models, and used directly for training and inferencing.

The APIs of converting to ONNX format are housed in hetu.onnx.

### Convert Hetu Models to ONNX Format

The following example shows how to convert the CNN model to the ONNX fomat.

#### Convert CNN Model to ONNX Format

The model is converted to ONNX format. Firstly, the operator of Hetu is mapped into the operator of ONNX one by one according to the operator protocol of ONNX, and then the diagram is built and converted to the model format and saved.

```python
from hetu import onnx as ax
#Build a CNN model using Hetu APIs.
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
z6 = ad.avg_pool2d_op(z5, kernel_H=2, kernel_W=2, padding=0, stride=2)m
z6_flat = ad.array_reshape_op(z6, (-1, 7 * 7 * 64))
y = ad.matmul_op(z6_flat, W3)+b3
executor=ad.Executor([y],ctx=executor_ctx)
onnx_input_path = 'hetu_cnn_model.onnx'
##################################
# Export a Hetu model into ONNX format.
#
#    Args:
#        executor: Hetu graph.
#        [X]: Input tensors in Hetu graph.
#        [y]: output tensors in Hetu graph.
#        onnx_input_path: a string for the onnx saving filename.
#
##################################
ax.hetu2onnx.export(executor,[X],[y],onnx_input_path)
```

#### **Inference using ONNXRuntime**

```python
import onnxruntime as rt
rand = np.random.RandomState(seed=123)
X_val=rand.normal(scale=0.1, size=(batch_size, 1,28,28)).astype(np.float32)
#Run inference using Hetu runtime.
ath= executor.run(feed_dict={X: X_val})
sess=rt.InferenceSession(onnx_input_path)
input=sess.get_inputs()[0].name
#Run inference using ONNXRuntime.
pre=sess.run(None,{input:X_val.astype(np.float32)})[0]
#Compare results using numpy testing API.
np.testing.assert_allclose(ath[0].asnumpy(),pre,rtol=1e-5)
```

#### **Convert ONNX Format to Hetu Model** 

This section takes the ONNX format of a CNN model transformation for TensorFlow as an example to show how to import the ONNX format into the Hetu framework.

### Convert Tensorflow Model to ONNX Format

Build TensorFlow model and convert to the ONNX format using TF2ONNX.

```python
import tensorflow.compat.v1 as tf
import tf2onnx
rand = np.random.RandomState(seed=123)
X_val = rand.normal(scale=0.1, size=(20, 784)).astype(np.float32)

#build a CNN model using TensorFlow APIs
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
  z9 = tf.reshape(z8, (-1, shape[0]))
  y = tf.matmul(z9, weight3)
  _ = tf.identity(y,name='output')
  sess.run(tf.global_variables_initializer())
  expected = sess.run(y, feed_dict={x: X_val})
  graph_def=tf.graph_util.convert_variables_to_constants(sess,sess.graph_def, 'output'])
tf.reset_default_graph()
tf.import_graph_def(graph_def, name='')
model_name='tf_cnn_model.onnx'
```

Then you can convert the model to ONXX format by the following code:

```python
with tf.Session() as sess:
  onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=['input:0'], output_names=['output:0'],)
  model_proto = onnx_graph.make_model('cnn_model')
  with open(model_name,'wb') as f:
    f.write(model_proto.SerializeToString())
```

#### **Convert ONNX Format to Hetu Model**

By converting the ONNX format into the Hetu framework, you can simply use hetu.onnx2hetu.load_onnx, and the resulting Executor can be used for subsequent training.

```python
x,y =  hx.onnx2hetu.load_onnx(model_name)
#build Hetu graph using ad.Executor.
executor=ad.Executor([y],ctx=ctx)
```

#### **Comparison of Conversion Results**

The ONNX protocol before the above transformation and the model after the transformation were inferred and compared with MNIST data set to verify the correctness of the transformation.ONNXRuntime is used for inferencing before conversion.

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

