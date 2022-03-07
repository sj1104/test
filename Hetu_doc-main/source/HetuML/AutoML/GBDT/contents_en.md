## GBDT

Here is a simple Python demo to show how to use AutoML module in HetuML for auto-tuning on Gradient Boosting Decision Tree(GBDT) in HetuML. 

Before tuning on the models, we need to define a search space for the current model, as well as an optimization function. 

### Define Hyper-parameters Search Space

Firstly, we utilize ConfigSpace library to define search space for hyper-parameters. We select a few hyper-parameters for tuning as an example, and you could easily change the tuning hyerper-parameters according to your need.  The example is as follows:

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

You can adaptively set hyper-parameters to its corresponding types(e.g., UniformFloatHyperparameter,  Constant).

### Define Optimization Function

Next, you could define optimizing function for this model, and in this example, here we set the optimizing objective to the minimization of log loss.

We then define the objective function. The train set is used in the objective function to train the GBDT model, and the validation set is used to predict: 

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

### Hyper-parameter Tuning

After the above steps, you could call SMBO for optimization. Here we set the maximal rounds of hyper-parameter tuning to 30, which represents that the maximal tuning times for HetuML Gradient Boosting Decision Tree is 30. And we set the time limit for each term of validation to 180 seconds, and the timeout task will be terminated.  

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

Finally, you could print out  the result：

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