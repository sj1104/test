## 快速入门

谢谢你选择Hetu！现在我们以一个单机版本的LR模型的训练来走进HetuML。

在开始之前，请确保你已经正确地安装了HetuML。

一般地，深度学习任务包含几个重要步骤：1.数据集加载；2.构建模型；3.模型训练；4.模型验证。下面你可以使用HetuML的API，来一步步实现上述步骤。

### 导入HetuML模块

新建一个python文件，并导入HetuML以及相关的模块。这里以单机版本的逻辑回归模型为例。

```python
from hetuml.linear import LogisticRegression
from hetuml.data import Dataset
import os
```

### 加载数据集

HetuML内置了数据集的加载和切分batch的方法，你可以使用以下方法加载数据集,其中，train_path、valid_path分别指向训练集和验证集的地址：

```python
# path to data
train_path = "./data/a9a/a9a"
valid_path = "./data/a9a/a9a.t"
# option 1: directly read from file
train_data = Dataset.from_file(train_path, neg_y=True)
valid_data = Dataset.from_file(train_path, neg_y=True)
```

除此之外，如果您的数据格式是libSVM格式，您也可以通过加载csr_matrix的方式加载数据：

```python
from sklearn.datasets import load_svmlight_file
X_train, y_train = load_svmlight_file(train_path)
y_train[y_train != 1] = -1
train_data = (X_train, y_train)
X_valid, y_valid = load_svmlight_file(valid_path)
y_valid[y_valid != 1] = -1
valid_data = (X_valid, y_valid)
```

### 构建模型

通过调用HetuML的api，可以方便的使用HetuML的模型进行训练与测试，我们以单机版的逻辑回归为例。详细模型可以被用户设置的超参可以在用户手册中查询。

首先，我们定义一些测试模型训练与验证的api：

```python
from sklearn.metrics import log_loss
def test_fit(model, train_data, valid_data):
    print(LOG_PREFIX + "Test fitting of " + model.name() + "..." + LOG_SUFFIX)
    model.fit(train_data, valid_data)
    print(LOG_PREFIX + "Test fitting of " + model.name() + " passed" + LOG_SUFFIX)
    
def test_predict(model, pred_data, labels=None):
    print(LOG_PREFIX + "Test prediction of " + model.name() + "..." + LOG_SUFFIX)
    pred = model.predict_proba(pred_data)
    if labels is not None:
        labels_cp = labels.copy()
        labels_cp[labels_cp == -1] = 0
        loss = log_loss(labels_cp, pred)
        print("Log loss: {:.6f}".format(loss))
    print(LOG_PREFIX + "Test prediction of " + model.name() + " passed" + LOG_SUFFIX)
    return pred

def test_evaluate(model, eval_data):
    print(LOG_PREFIX + "Test evaluation of " + model.name() + "..." + LOG_SUFFIX)
    metrics = model.evaluate(eval_data, ["log-loss", "error", "precision"])
    print("eval metrics: {}".format(metrics))
    print(LOG_PREFIX + "Test evaluation of " + model.name() + " passed" + LOG_SUFFIX)
```

然后，我们定义模型并且使用模型：

```python
lr = LogisticRegression(learning_rate=0.5, metrics="log-loss,error,precision")
test_fit(lr, train_data, valid_data)
test_predict(loaded, valid_data)
test_evaluate(loaded, valid_data)
```



### 结果输出

对于以上单机版本的逻辑回归模型的训练和推理，在训练过程中你会得到与以下格式类似的log：

```python
[2021-06-07 08:02:29:432 (io.h:117)] [INFO] *HETU* Read libsvm data from /home/xiaonan/jyzh/libsvm_data/a9a cost 427 ms
[2021-06-07 08:02:29:873 (io.h:117)] [INFO] *HETU* Read libsvm data from /home/xiaonan/jyzh/libsvm_data/a9a cost 439 ms
 ********** Test fitting of LogisticRegression... **********
[2021-06-07 08:02:29:874 (linear.h:132)] [INFO] *HETU* Start to fit LogisticRegression model with hyper-parameters:
|NUM_EPOCH = 10
|BATCH_SIZE = 1000
|LEARNING_RATE = 0.5
|L1_REG = 0.0
|L2_REG = 0.0
|LOSS = logistic
|METRICS = log-loss,error,precision

[2021-06-07 08:02:29:878 (linear.h:140)] [INFO] *HETU* Epoch[0] Train loss[0.430527]
[2021-06-07 08:02:29:928 (linear.h:144)] [INFO] *HETU* Epoch[0] Valid log-loss[0.376146] error[0.172016] precision[0.827984]
[2021-06-07 08:02:29:933 (linear.h:140)] [INFO] *HETU* Epoch[1] Train loss[0.364551]
[2021-06-07 08:02:29:934 (linear.h:144)] [INFO] *HETU* Epoch[1] Valid log-loss[0.355389] error[0.164676] precision[0.835324]
[2021-06-07 08:02:29:938 (linear.h:140)] [INFO] *HETU* Epoch[2] Train loss[0.350699]
[2021-06-07 08:02:29:939 (linear.h:144)] [INFO] *HETU* Epoch[2] Valid log-loss[0.346139] error[0.161819] precision[0.838181]
[2021-06-07 08:02:29:944 (linear.h:140)] [INFO] *HETU* Epoch[3] Train loss[0.343606]
[2021-06-07 08:02:29:945 (linear.h:144)] [INFO] *HETU* Epoch[3] Valid log-loss[0.340778] error[0.160069] precision[0.839931]
[2021-06-07 08:02:29:949 (linear.h:140)] [INFO] *HETU* Epoch[4] Train loss[0.33927]
[2021-06-07 08:02:29:950 (linear.h:144)] [INFO] *HETU* Epoch[4] Valid log-loss[0.337316] error[0.15884] precision[0.84116]
[2021-06-07 08:02:29:954 (linear.h:140)] [INFO] *HETU* Epoch[5] Train loss[0.336377]
[2021-06-07 08:02:29:955 (linear.h:144)] [INFO] *HETU* Epoch[5] Valid log-loss[0.334919] error[0.157428] precision[0.842572]
[2021-06-07 08:02:29:959 (linear.h:140)] [INFO] *HETU* Epoch[6] Train loss[0.334327]
[2021-06-07 08:02:29:960 (linear.h:144)] [INFO] *HETU* Epoch[6] Valid log-loss[0.333174] error[0.156414] precision[0.843586]
[2021-06-07 08:02:29:965 (linear.h:140)] [INFO] *HETU* Epoch[7] Train loss[0.332807]
[2021-06-07 08:02:29:966 (linear.h:144)] [INFO] *HETU* Epoch[7] Valid log-loss[0.331852] error[0.155984] precision[0.844016]
[2021-06-07 08:02:29:970 (linear.h:140)] [INFO] *HETU* Epoch[8] Train loss[0.33164]
[2021-06-07 08:02:29:971 (linear.h:144)] [INFO] *HETU* Epoch[8] Valid log-loss[0.330819] error[0.155646] precision[0.844354]
[2021-06-07 08:02:29:975 (linear.h:140)] [INFO] *HETU* Epoch[9] Train loss[0.33072]
[2021-06-07 08:02:29:976 (linear.h:144)] [INFO] *HETU* Epoch[9] Valid log-loss[0.329992] error[0.15537] precision[0.84463]
[2021-06-07 08:02:29:976 (linear.h:149)] [INFO] *HETU* Fit LogisticRegression model cost 102 ms
 ********** Test fitting of LogisticRegression passed **********
 ********** Test saving & loading of LogisticRegression... **********
[2021-06-07 08:02:29:983 (mlbase.h:94)] [INFO] *HETU* Save LogisticRegression model to ./test_models/LogisticRegression done
[2021-06-07 08:02:29:984 (mlbase.h:84)] [INFO] *HETU* Load LogisticRegression model from ./test_models/LogisticRegression done
 ********** Test saving & loading of LogisticRegression passed **********
 ********** Test prediction of LogisticRegression... **********
 ********** Test prediction of LogisticRegression passed **********
 ********** Test evaluation of LogisticRegression... **********
eval metrics: {'log-loss': 0.3299917, 'error': 0.15536992, 'precision': 0.84463006}
 ********** Test evaluation of LogisticRegression passed **********
```