import numpy as np
import pandas as pd
from decision_tree import tree_node, make_test_y_values1, make_test_y_values2


class boosted:
    def __init__(self,
                 x_data: type(pd.DataFrame) = pd.DataFrame(),
                 y_data: np.ndarray = np.array([]),
                 weights: np.ndarray = np.array([]),
                 max_num_trees: int = 100,
                 tolerance: float = .01,
                 stop_depth: int = 5,
                 min_points_in_split: int = 1,
                 tree_type: str = 'classification',
                 classification_impurity: str = 'gini_index',
                 use_random_features: bool = False,
                 num_random_features: int = 0):
        self.trees = {}
        if type(x_data) != type(None):
            self.train(x_data=x_data,
                       y_data=y_data,
                       weights=weights,
                       max_num_trees=max_num_trees,
                       tolerance=tolerance,
                       stop_depth=stop_depth,
                       min_points_in_split=min_points_in_split,
                       tree_type=tree_type,
                       classification_impurity=classification_impurity,
                       use_random_features=use_random_features,
                       num_random_features=num_random_features)
    def train(self,
              x_data: type(pd.DataFrame) = pd.DataFrame(),
              y_data: np.ndarray = np.array([]),
              weights: np.ndarray = np.array([]),
              max_num_trees: int = 100,
              tolerance: float = .01,
              stop_depth: int = 5,
              min_points_in_split: int = 1,
              tree_type: str = 'classification',
              classification_impurity: str = 'gini_index',
              use_random_features: bool = False,
              num_random_features: int = 0):
        if len(weights) == 0:
            weights = np.ones(len(y_data))/len(y_data)
        x_data_copy = x_data.copy()
        y_data_copy = y_data.copy()
        for n in range(max_num_trees):
            tree = tree_node(x_data=x_data_copy,
                             y_data=y_data_copy,
                             stop_depth=stop_depth,
                             min_points_in_split=min_points_in_split,
                             tree_type=tree_type,
                             classification_impurity=classification_impurity,
                             use_random_features=use_random_features,
                             num_random_features=num_random_features)
            diff = np.ones(y_data.shape)
            y_pred = tree.evaluate_many(x_data)
            diff[y_pred-y_data == 0] = 0
            weighted_diff_sum = (diff*weights).sum()
            weight_sum = weights.sum()
            error = weighted_diff_sum/weight_sum
            if error <= 0:
                alpha_n = 100000000000000
            elif error >= 1:
                alpha_n = -10000000000000
            else:
                alpha_n = 0.5*np.log((1-error)/(error))
            self.trees[n] = {'tree':tree, 'alpha':alpha_n}
            weights = weights*np.exp(-alpha_n*y_pred*y_data)
            distribution = weights/weights.sum()
            x_data_copy, y_data_copy = self.new_distribution(x_data,y_data,distribution, 100)
            y_pred = self.evaluate_many(x_data)
            diff = np.ones(y_data.shape)
            diff[y_pred-y_data == 0] = 0
            if n % 10 == 9:
                print(f'    num wrong: {diff.sum()}/{len(diff)} ({diff.sum()/len(diff)*100:.2f}%)')
                print(f'    num trees: {len(self.trees)}\n')
            if diff.sum()/len(diff)< tolerance:
                break

    def evaluate_many(self,
                      x_data: type(pd.DataFrame([]))
                      )->np.ndarray:
        results = []
        df1 = x_data.copy()
        for index in self.trees:
            tree = self.trees[index]['tree']
            alpha = self.trees[index]['alpha']
            # print(len(df1))
            results.append(alpha*tree.evaluate_many(df1))
        results = np.array(results)
        results = results.sum(axis = 0)
        results[results >= 0] = 1
        results[results < 0] = -1
        return results

    def new_distribution(self,
                         x_data: type(pd.DataFrame([])),
                         y_data: np.ndarray,
                         weights: np.ndarray,
                         new_set_size: int = 0
                         )->(type(pd.DataFrame([])), np.ndarray):
        x_data = x_data.copy()
        y_data = y_data.copy()
        if new_set_size <= 0:
            new_set_size = len(x_data)
        choices = [i for i in range(len(x_data))]
        indices = np.random.choice(choices, size = new_set_size, p=weights)
        new_x = x_data.loc[indices]
        new_y = y_data[indices]
        new_x = new_x.reset_index(drop=True)
        return new_x,new_y


if __name__ == '__main__':
    df1 = pd.DataFrame({'f1':[i for i in range(100)],
                        'f2':[0 for i in range(100)]})
    df2 = pd.DataFrame({'f1':[i for i in range(100)],
                        'f2':[1 for i in range(100)]})
    x_data_test = pd.concat([df1, df2]).reset_index(drop=True)
    y_data_test = x_data_test.apply(lambda row:make_test_y_values1(row.f1)*make_test_y_values2(row.f2), axis=1).to_numpy()

    x_val = pd.DataFrame({'f1':[np.random.randint(0,100) for i in range(200)],
                        'f2':[np.random.randint(0,2) for i in range(200)]})
    y_val = x_val.apply(lambda row:make_test_y_values1(row.f1)*make_test_y_values2(row.f2), axis=1).to_numpy()

    x_data = pd.DataFrame({'f1':[np.random.randint(0,100) for i in range(300)],
                        'f2':[np.random.randint(0,2) for i in range(300)]})
    y_data = x_data.apply(lambda row:make_test_y_values1(row.f1)*make_test_y_values2(row.f2), axis=1).to_numpy()
    boost = boosted(x_data=x_data,
                    y_data=y_data,
                    weights=None,
                    max_num_trees=200,
                    tolerance=.001,
                    stop_depth=3,
                    min_points_in_split=5,
                    tree_type='classification',
                    classification_impurity='gini_index',
                    use_random_features=True,
                    num_random_features=0)

    y_pred = boost.evaluate_many(x_val)
    print('val evaluation')
    diff = np.ones(y_val.shape)
    diff[y_pred-y_val == 0] = 0
    print(f'num wrong: {diff.sum()}/{len(diff)} ({diff.sum()/len(diff)*100:.2f}%)')

    y_pred = boost.evaluate_many(x_data_test)
    print('test evaluation')
    diff = np.ones(y_data_test.shape)
    diff[y_pred-y_data_test == 0] = 0
    print(f'num wrong: {diff.sum()}/{len(diff)} ({diff.sum()/len(diff)*100:.2f}%)')
    print(f'num trees: {len(boost.trees)}')
