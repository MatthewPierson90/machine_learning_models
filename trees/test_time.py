from time import perf_counter as tt
from decision_tree import tree_node, make_test_y_values1, make_test_y_values2
from decision_tree_with_cython import tree_node as ctree_node
import numpy as np
import pandas as pd

df1 = pd.DataFrame({'f1':[i for i in range(100)],
                    'f2':[0 for i in range(100)]})
df2 = pd.DataFrame({'f1':[i for i in range(100)],
                    'f2':[1 for i in range(100)]})
x_test = pd.concat([df1, df2]).reset_index(drop=True)
y_test = x_test.apply(lambda row:make_test_y_values1(row.f1)*make_test_y_values2(row.f2), axis=1).to_numpy()

x_data = pd.DataFrame({'f1':[np.random.randint(0,100) for i in range(100)],
                    'f2':[np.random.randint(0,2) for i in range(100)]})
y_data = x_data.apply(lambda row:make_test_y_values1(row.f1)*make_test_y_values2(row.f2), axis=1).to_numpy()

time1 = 0
train_time1 = 0
eval_time1 = 0
for k in range(100):
    s = tt()
    node = tree_node(stop_depth=20,
                     current_depth=0,
                     x_data=x_data.copy(),
                     y_data=y_data.copy(),
                     min_points_in_split=1,
                     tree_type='regression',
                     classification_impurity='gini_index',
                     use_random_features=False,
                     num_random_features=0)
    train_time1 += tt()-s
    if k==99:
        s1 = tt()
        y_pred_train = node.evaluate_many(x_data.copy())
        print('train evaluation')
        diff = np.ones(y_data.shape)
        diff[y_pred_train-y_data == 0] = 0
        print(f'num wrong: {diff.sum()}/{len(diff)} ({diff.sum()/len(diff)*100:.2f}%)')

        y_pred = node.evaluate_many(x_test.copy())
        print('test evaluation')
        diff = np.ones(y_test.shape)
        diff[y_pred-y_test == 0] = 0
        print(f'num wrong: {diff.sum()}/{len(diff)} ({diff.sum()/len(diff)*100:.2f}%)')
        eval_time1 += tt()-s1
    time1 += tt()-s
print(time1)
print(train_time1)
print(eval_time1)

time2 = 0
train_time2 = 0
eval_time2 = 0
for k in range(100):
    s = tt()
    node = ctree_node(stop_depth=20,
                     current_depth=0,
                     x_data=x_data,
                     y_data=y_data,
                     min_points_in_split=1,
                     tree_type='regression',
                     classification_impurity='gini_index',
                     use_random_features=False,
                     num_random_features=0)
    train_time2 += tt()-s
    if k == 99:
        s1 = tt()
        y_pred_train = node.evaluate_many(x_data)
        print('train evaluation')
        diff = np.ones(y_data.shape)
        diff[y_pred_train-y_data == 0] = 0
        print(f'num wrong: {diff.sum()}/{len(diff)} ({diff.sum()/len(diff)*100:.2f}%)')

        y_pred = node.evaluate_many(x_test)
        print('test evaluation')
        diff = np.ones(y_test.shape)
        diff[y_pred-y_test == 0] = 0
        print(f'num wrong: {diff.sum()}/{len(diff)} ({diff.sum()/len(diff)*100:.2f}%)')
        eval_time2 += tt()-s1
    time2 += tt()-s
print(time2)
print(train_time2)
print(eval_time2)
