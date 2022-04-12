import numpy as np
import pandas as pd
from time import perf_counter
tt = perf_counter
from decision_tree import tree_node, make_test_y_values1, make_test_y_values2


class random_forest:
    def __init__(self,
                 num_trees: int = 10,
                 stop_depth: int = 5,
                 x_data: type(pd.DataFrame) = pd.DataFrame(),
                 y_data: np.ndarray = np.array([]),
                 min_points_in_split: int = 1,
                 tree_type: str = 'classification',
                 classification_impurity: str = 'gini_index',
                 use_bootstrap: bool = True,
                 new_set_size: int = 0,
                 use_random_features: bool = False,
                 num_random_features: int = 0
                 ) -> None:

        self.trees = [tree_node(stop_depth=stop_depth,
                                current_depth=0,
                                min_points_in_split=min_points_in_split,
                                tree_type=tree_type,
                                classification_impurity=classification_impurity,
                                use_random_features=use_random_features,
                                num_random_features=num_random_features) for _ in range(num_trees)]
        self.was_trained = False
        if len(x_data) != 0:
            self.train(x_data,
                       y_data,
                       min_points_in_split=min_points_in_split,
                       tree_type=tree_type,
                       classification_impurity=classification_impurity,
                       use_random_features=use_random_features,
                       num_random_features=num_random_features,
                       use_bootstrap=use_bootstrap,
                       new_set_size=new_set_size,)


    def make_bootstrap_data(self,
                            x_data: type(pd.DataFrame) = pd.DataFrame(),
                            y_data: np.ndarray = np.array([]),
                            new_set_size: int = 0):
        x_data = x_data.copy()
        y_data = y_data.copy()
        if new_set_size <= 0:
            new_set_size = len(x_data)

        rand_indices = np.random.randint(0, len(x_data), new_set_size)
        new_x = x_data.loc[rand_indices]
        new_y = y_data[rand_indices]
        new_x = new_x.reset_index(drop=True)
        return new_x,new_y

    def train(self,
              x_data: type(pd.DataFrame) = pd.DataFrame(),
              y_data: np.ndarray = np.array([]),
              min_points_in_split: int = 1,
              tree_type: str = 'classification',
              classification_impurity: str = 'gini_index',
              use_bootstrap: bool = True,
              new_set_size: int = 0,
              use_random_features: bool = False,
              num_random_features: int = 0
              ) -> None:
        self.was_trained = True
        self.features = x_data.columns
        self.tree_type = tree_type
        for tree in self.trees:
            if use_bootstrap:
                x_data_boot, y_data_boot = self.make_bootstrap_data(x_data, y_data, new_set_size)
            else:
                x_data_boot = x_data.copy()
                y_data_boot = y_data.copy()
            tree.train(x_data=x_data_boot,
                       y_data=y_data_boot,
                       min_points_in_split=min_points_in_split,
                       tree_type=tree_type,
                       classification_impurity=classification_impurity,
                       use_random_features=use_random_features,
                       num_random_features=num_random_features)


    def evaluate_single(self,
                        single_x_data):
        if self.was_trained:
            if type(single_x_data) == type(np.array(1)) or type(single_x_data) == type([]):
                single_x_data = np.array(single_x_data)
                if single_x_data.shape == (len(self.features),):
                    x_data = pd.DataFrame([single_x_data], columns = self.features)
                elif single_x_data.shape == (1,len(self.features)):
                    x_data = pd.DataFrame(single_x_data, columns=self.features)
                else:
                    print('incorrect data shape')
                    raise TypeError
            elif type(single_x_data) == type(pd.DataFrame()):
                x_data = single_x_data
            else:
                print('incorrect data type, use pandas DataFrame, numpy array, or list')
                raise TypeError
            results = []
            for tree in self.trees:
                results.append(tree.evaluate_single(x_data))
            if self.tree_type == 'regression':
                return np.array(results).mean()
            else:
                vals, counts = np.unique(np.array(results), return_counts=True)
                index = np.argmax(counts)
                return vals[index]
        else:
            print('Tree must first be trained!')

    def evaluate_many(self,
                      df: type(pd.DataFrame())
                      ) -> type(np.array([])):
        results = []
        df1 = df.copy()
        for tree in self.trees:
            results.append(tree.evaluate_many(df1))
        ar = np.array(results)
        N = ar.shape[1]
        res = []
        for n in range(N):
            vals, counts = np.unique(ar[:, n], return_counts=True)
            index = counts.argmax()
            res.append(vals[index])
        return np.array(res)


if __name__ == '__main__':
    df1 = pd.DataFrame({'f1':[i for i in range(100)],
                        'f2':[0 for i in range(100)]})
    df2 = pd.DataFrame({'f1':[i for i in range(100)],
                        'f2':[1 for i in range(100)]})
    df = pd.concat([df1,df2]).reset_index(drop=True)
    y_vals = df.apply(lambda row: make_test_y_values1(row.f1)+make_test_y_values2(row.f2), axis=1).to_numpy()

    start = tt()
    forest = random_forest(num_trees=10,
                           stop_depth=10,
                           x_data=df,
                           y_data=y_vals,
                           use_bootstrap=True,
                           new_set_size=200,
                           min_points_in_split=1,
                           tree_type='classification',
                           classification_impurity='gini_index',
                           use_random_features=True,
                           num_random_features=2)
    end = tt()
    print(f'Training time: {end-start}')
    start = tt()
    r = forest.evaluate_many(df)
    end = tt()
    print(f'eval time: {end-start}')
    print(r)
    # ar = np.array(r)
    # results = []
    # for n in range(ar.shape[1]):
    #     vals, counts = np.unique(ar[:,n], return_counts=True)
    #     index = counts.argmax()
    #     results.append(vals[index])
    # print(results)
    # index = np.argmax(counts,axis=1)
    # results = []
    # for index in range(200):
    #     results.append(forest.evaluate_single(df.loc[[index]]))
    # r = np.array(results)
    # # print(r)
    # diff = np.ones(200)[(r-y_vals) != 0]
    # print(diff.sum()/2)