## GBDT

本⽂演示如何使⽤Hetu自动化机器学习模块对HetuML系统中的梯度提升决策树模型进行自动化调参。

在进⾏自动化调参之前，我们需要定义任务搜索空间（即超参数空间）和优化⽬标函数。

### 定义超参数空间

首先，我们我们使⽤ConfigSpace库定义模型的超参数空间。我们介绍HetuML的梯度提升决策树模型中可以被定义为超参的参数，并定义它们在Hetu自动化机器学习模块中对应的超参类型。

作为示例，我们选择部分参数作为超参训练:

```python
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import UniformFloatHyperparameter, Constant,\
UniformIntegerHyperparameter
def get_config_space():
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3,
    default_value=0.1, log=True)
    num_round = UniformIntegerHyperparameter("num_round", 1, 100, default_value=10)
    max_depth = UniformIntegerHyperparameter("max_depth", 1, 32, default_value=6)
    ins_sp_ratio = UniformFloatHyperparameter("ins_sp_ratio", 0, 1,
    default_value=1, log=False)
    feat_sp_ratio = UniformFloatHyperparameter("feat_sp_ratio", 0, 1,
    default_value=1, log=False)
    cs.add_hyperparameters([learning_rate, num_round, max_depth, ins_sp_ratio, feat_sp_ratio])
    return cs
```

根据模型的不同与参数类型的不同，也可以相应选择Constant设置常量、UniformIntegerHyperparameter为Int型参数进行优化。

### 定义目标函数

接下来，我们定义优化⽬标函数，设置最小化log loss为优化目标。

下面展示我们定义目标函数objective_function,并在目标函数中利用训练集训练HetuML提供的GBDT模型，利用验证集预测并计算优化log loss的过程:

```python
from hetuml.ensemble import GBDT
from sklearn.datasets import load_svmlight_file

config_space = get_config_space()
X_train, y_train = load_svmlight_file(train_path)
y_train[y_train != 1] = 0
train_data = (X_train, y_train)
X_valid, y_valid = load_svmlight_file(valid_path)
y_valid[y_valid != 1] = 0
valid_data = (X_valid, y_valid)

def objective_function(config):
    # convert Configuration to dict
    params = config.get_dictionary()
    learning_rate=params.get('learning_rate',0.1)
    num_round = params.get('num_round',10)
    max_depth=params.get('max_depth',6)
    ins_sp_ratio = params.get('ins_sp_ratio',1)
    feat_sp_ratio = params.get('feat_sp_ratio',1)
    model = GBDT(num_round=num_round,ins_sp_ratio=ins_sp_ratio,feat_sp_ratio=feat_sp_ratio,learning_rate=learning_rate, max_depth=max_depth, metrics="log-loss,error,precision")
    
    model.fit(train_data, valid_data)
    metrics = model.evaluate(valid_data, ["log-loss", "error", "precision"])
    print("eval metrics: {}".format(metrics))
    
    log_loss = metrics["log-loss"]   
    result = dict(objs=(log_loss, ))
    print(result)
    return result
```

### 执行优化

定义好任务和⽬标函数以后，就可以调⽤自动化⻉叶斯优化框架SMBO执⾏优化。我们设置优化轮数（max_runs）为30，代表将对GBDT模型调参30轮。每轮最⼤验证时间（time_limit_per_trial）设置为180秒，超时的任务将被终⽌。优化结束后，可以打印优化结果。

```python
from openbox.optimizer.generic_smbo import SMBO
bo = SMBO(objective_function,
config_space,
max_runs=30,
time_limit_per_trial=180,
task_id='tuning_gbdt')
history = bo.run()
print(history)
```

打印结果如下：

```python
+-----------------------------------------+
| Parameters              | Optimal Value |
+-------------------------+---------------+
| feat_sp_ratio           | 0.776215      |
| ins_sp_ratio            | 0.479199      |
| learning_rate           | 0.080059      |
| max_depth               | 13            |
| num_round               | 100           |
+-------------------------+---------------+
| Optimal Objective Value | 0.32175794    |
+-------------------------+---------------+
| Num Configs             | 27            |
+-------------------------+---------------+
```