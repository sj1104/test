## LR

Here is a simple Python demo to show how to use AutoML module in HetuML for auto-tuning on Logistic Regression(LR) in HetuML. 

Before tuning on the models, we need to define a search space for the current model, as well as an optimization function. 

###  Define Hyper-parameters Search Space

Firstly, we utilize ConfigSpace library to define search space for hyper-parameters. Parameter mainly need to be tuned in Logistic Regression is learning rate.  The example is as follows:

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

You can adaptively set hyper-parameters to its corresponding types(e.g., UniformFloatHyperparameter,  Constant).

### Define Optimization Function

Next, you could define optimizing function for this model, and in this example, we set the optimizing objective to the minimization of log loss.

We then define the objective function. The train set is used in the objective function to train the LR model, and the validation set is used to predict: 

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

### Hyper-parameter Tuning

After the above steps, you could call SMBO for optimization. Here we set the maximal rounds of hyper-parameter tuning to 30, which represents that the maximal tuning times for HetuML Logistic Regression is 30. And we set the time limit for each term of validation to 180 seconds, and the timeout task will be terminated.  

```python
from openbox.optimizer.generic_smbo import SMBO
bo = SMBO(objective_function,
    config_space,
    max_runs=30,
    time_limit_per_trial=180,
    task_id='tuning_lr')
history = bo.run()
```

Finally, you could print out  the result：

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