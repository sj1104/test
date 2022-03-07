## LR

HetuML是一个高性能的分布式机器学习框架，并且支持自动化调参。我们通过使用OpenBox对HetuML中的模型进行超参数优化，并演示HetuML进行自动化调参的过程。

在进⾏自动化调参之前，我们需要定义任务搜索空间（即超参数空间）和优化⽬标函数。

### 定义超参数空间

首先，我们我们使⽤ConfigSpace库定义模型的超参数空间。在逻辑回归中，需要进行自动化调参的参数主要是学习率:

```python
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import UniformFloatHyperparameter, Constant,\
UniformIntegerHyperparameter
def get_config_space():
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3,
    default_value=0.1, log=True)
    cs.add_hyperparameters([learning_rate])
    return cs
config_space = get_config_space()
```

根据模型的不同与参数类型的不同，也可以相应选择Constant设置常量、UniformIntegerHyperparameter为Int型参数进行优化。

### 定义目标函数

接下来，我们定义优化⽬标函数，设置最小化log loss为优化目标。

下面展示我们定义目标函数objective_function,并在目标函数中利用训练集训练HetuML提供的LogisticRegression模型，利用验证集预测并计算优化log loss的过程:

```python
from hetuml.linear import LogisticRegression
from sklearn.datasets import load_svmlight_file

X_train, y_train = load_svmlight_file(train_path)
y_train[y_train != 1] = -1
train_data = (X_train, y_train)
X_valid, y_valid = load_svmlight_file(valid_path)
y_valid[y_valid != 1] = -1
valid_data = (X_valid, y_valid)

def objective_function(config):
    # convert Configuration to dict
    params = config.get_dictionary()
    learning_rate = params.get('learning_rate', 0.1)
    model = LogisticRegression(learning_rate=learning_rate, metrics="log-loss,error,precision")
    model.fit(train_data, valid_data)
    metrics = model.evaluate(valid_data, ["log-loss", "error", "precision"])
    print("Eval metrics: {}".format(metrics))
    loss = metrics['log-loss']
    result = dict(objs=(loss, ))
    return result
```

### 执行优化

定义好任务和⽬标函数以后，就可以调⽤自动化⻉叶斯优化框架SMBO执⾏优化。我们设置优化轮数（max_runs）为30，代表将对LogisticRegression模型调参30轮。每轮最⼤验证时间（time_limit_per_trial）设置为180秒，超时的任务将被终⽌。优化结束后，可以打印优化结果。

```python
from openbox.optimizer.generic_smbo import SMBO
bo = SMBO(objective_function,
    config_space,
    max_runs=30,
    time_limit_per_trial=180,
    task_id='tuning_lr')
history = bo.run()
```

打印结果如下：

```python
print(history)
+-----------------------------------------+
| Parameters              | Optimal Value |
+-------------------------+---------------+
| learning_rate           | 0.268633      |
+-------------------------+---------------+
| Optimal Objective Value | 0.33166483    |
+-------------------------+---------------+
| Num Configs             | 30            |
+-------------------------+---------------+
```