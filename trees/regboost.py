from decision_tree import tree_node
import pandas as pd
import numpy as np

class boosted_reg:
    def __init__(self,
                 stop_depth_boosted: int = 5,
                 stop_depth_initial: int = 5,
                 x_data: type(pd.DataFrame()) = pd.DataFrame(),
                 y_data: np.ndarray = np.array([]),
                 max_trees: int = 200,
                 tolerance: float = 0.001,
                 learning_rate: float = 0.1,
                 regularization_constant: float = 0,
                 node_penalty: float = 0,
                 min_points_in_split: int = 1,
                 use_random_features: bool = False,
                 num_random_features: int = 0
                 )->None:
        self.trees = {}
        self.num_trees=0
        if len(x_data) != 0:
            self.train(stop_depth_boosted=stop_depth_boosted,
                       stop_depth_initial=stop_depth_initial,
                       x_data=x_data,
                       y_data=y_data,
                       max_trees=max_trees,
                       tolerance=tolerance,
                       learning_rate=learning_rate,
                       regularization_constant=regularization_constant,
                       node_penalty=node_penalty,
                       min_points_in_split=min_points_in_split,
                       use_random_features=use_random_features,
                       num_random_features=num_random_features)

    def train(self,
              stop_depth_boosted: int = 5,
              stop_depth_initial: int = 5,
              x_data: type(pd.DataFrame()) = pd.DataFrame(),
              y_data: np.ndarray = np.array([]),
              max_trees: int = 200,
              tolerance: float = 0.001,
              learning_rate: float = 0.1,
              regularization_constant: float = 0,
              node_penalty: float = 0,
              min_points_in_split: int = 1,
              use_random_features: bool = False,
              num_random_features: int = 0
              )->None:
        self.features = x_data.columns
        residuals = y_data.copy()
        mse_old = np.inf
        for n in range(max_trees):
            if len(self.trees) == 0:
                tree = tree_node(stop_depth=stop_depth_initial,
                                 x_data=x_data,
                                 y_data=residuals,
                                 min_points_in_split=min_points_in_split,
                                 tree_type='regression',
                                 node_penalty=node_penalty,
                                 use_random_features=use_random_features,
                                 num_random_features=num_random_features)
                self.trees[n] = {'tree': tree,'learning_rate':1}
                y_pred = self.evaluate_many(x_data)
                residuals = y_data-y_pred
                mse = (residuals**2).sum()/(len(residuals))
                # print(y_pred)
                # print(residuals)
                # print('\n')
            else:
                tree = tree_node(stop_depth=stop_depth_boosted,
                                 x_data=x_data,
                                 y_data=residuals,
                                 min_points_in_split=min_points_in_split,
                                 tree_type='regression',
                                 is_boosted=True,
                                 regularization_constant=regularization_constant,
                                 node_penalty=node_penalty,
                                 use_random_features=use_random_features,
                                 num_random_features=num_random_features)
                self.trees[n] = {'tree':tree, 'learning_rate':learning_rate}
                y_pred = self.evaluate_many(x_data)
                residuals = y_data-y_pred
            mse = (residuals**2).sum()/(len(residuals))
            if n%10==9:
                print(f'num trees: {n+1}')
                print(f'current MSE: {mse}\n')
            if mse < tolerance or abs(mse_old-mse)==0:
                break
            mse_old = mse
        print(f'num trees: {n+1}')
        print(f'current MSE: {mse}\n')


    def evaluate_many(self,
                      x_data: type(pd.DataFrame())
                      )->np.ndarray:
        results = np.zeros(len(x_data), dtype=float)
        for tree_index in self.trees:
            tree = self.trees[tree_index]['tree']
            learning_rate = self.trees[tree_index]['learning_rate']
            eval = tree.evaluate_many(x_data)
            results += learning_rate*eval
        return results





if __name__ == '__main__':
    from matplotlib import pyplot as plt
    x_vals1 = np.random.rand(5000)*10
    x_vals2 = np.random.rand(5000)*10-5
    x_train = pd.DataFrame({'f1':x_vals1, 'f2':x_vals2})
    y_train = (x_vals2/5)*x_vals1**2
    boosted = boosted_reg(stop_depth_initial=2,
                          stop_depth_boosted=5,
                          x_data=x_train,
                          y_data=y_train,
                          max_trees=100,
                          tolerance=0.1,
                          learning_rate=.1,
                          regularization_constant=.5,
                          node_penalty=5,
                          min_points_in_split=1,
                          use_random_features=False,
                          num_random_features=0)
    x_1 = np.linspace(0, 10, 100)
    x_2 = np.linspace(-5, 5, 100)
    xx1, xx2 = np.meshgrid(x_1, x_2)
    data = pd.DataFrame({'f1':xx1.flatten(),'f2':xx2.flatten()})
    y = boosted.evaluate_many(data)
    yy = y.reshape(100,100)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx1, xx2, yy, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    # x_test_vals = np.linspace(0,10,1000)
    # x_test = pd.DataFrame({'f':x_test_vals})
    # y_test = x_test_vals**2
    # y_pred = boosted.evaluate_many(x_test)
    # plt.plot(x_test, y_pred)
    # print(f'test mse: {((y_pred-y_test)**2).sum()/len(y_test)}')


