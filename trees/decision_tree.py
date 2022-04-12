import numpy as np
import pandas as pd

class tree_node(object):
    """
    The basic node of a decision tree. Can be used to build a decision tree.
    This class is used for all the other classes (eg random forests and boosted trees).
    """
    def __init__(self,
                 stop_depth: int = 5,
                 current_depth: int = 0,
                 x_data: "pd.DataFrame" = pd.DataFrame(),
                 y_data:np.ndarray = np.array([]),
                 min_points_in_split: int = 1,
                 tree_type: str = 'classification',
                 classification_impurity: str ='gini_index',
                 is_boosted: bool = False,
                 regularization_constant: float = 0,
                 node_penalty: float = 0,
                 use_random_features: bool = False,
                 num_random_features: int = 0,
                 name: str = '',
                 node_dict: dict = {1:1})->None:
        """
        Makes a new tree node, automatically builds a tree if data is passed as an input.
        The name and node_dict are stored to get a picture of the tree easily.
        stop_depth: int = 5,
        current_depth: int = 0,
        x_data: type(pd.DataFrame) = pd.DataFrame(),
        y_data:np.ndarray = np.array([]),
        min_points_in_split: int = 1,
        tree_type: str = 'classification',
        classification_impurity: str ='gini_index',
        is_boosted: bool = False,
        regularization_constant: float = 0,
        node_penalty: float = 0,
        use_random_features: bool = False,
        num_random_features: int = 0,
        name: str = '',
        node_dict: dict = {1:1}
        """
        if current_depth == 0:
            node_dict = {}
        self.name = name
        node_dict[name] = self
        self.node_dict = node_dict
        self.was_trained = False
        self.is_leaf = False
        self.current_depth = current_depth
        self.stop_depth = stop_depth
        if len(x_data) != 0:
            self.train(x_data=x_data,
                       y_data=y_data,
                       min_points_in_split=min_points_in_split,
                       tree_type=tree_type,
                       classification_impurity=classification_impurity,
                       is_boosted=is_boosted,
                       regularization_constant=regularization_constant,
                       node_penalty=node_penalty,
                       use_random_features=use_random_features,
                       num_random_features=num_random_features)

    def train(self,
              x_data: type(pd.DataFrame) = pd.DataFrame(),
              y_data: np.ndarray = np.array([]),
              min_points_in_split: int = 1,
              tree_type: str = 'classification',
              classification_impurity: str = 'gini_index',
              is_boosted: bool = False,
              regularization_constant: float = 0,
              node_penalty: float = 0,
              use_random_features: bool = False,
              num_random_features: int = 0
              )->None:
        """
        Trains the node.

        stop_depth: int = 5,
        current_depth: int = 0,
        x_data: type(pd.DataFrame) = pd.DataFrame(),
        y_data:np.ndarray = np.array([]),
        min_points_in_split: int = 1,
        tree_type: str = 'classification',
        classification_impurity: str ='gini_index',
        is_boosted: bool = False,
        regularization_constant: float = 0,
        node_penalty: float = 0,
        use_random_features: bool = False,
        num_random_features: int = 0,
        """
        self.was_trained = True
        self.tree_type = tree_type
        self.min_points_in_split = min_points_in_split
        self.classification_impurity = classification_impurity
        self.features = x_data.columns
        if use_random_features:
            self.random_features = []
            feats = list(self.features).copy()
            if num_random_features == 0:
                num_random_features = int(np.sqrt(len(self.features)))
            for n in range(min([num_random_features, len(self.features)])):
                feat = np.random.choice(feats)
                self.random_features.append(feat)
                feats.remove(feat)
        else:
            self.random_features = self.features
        if self.current_depth == self.stop_depth or len(y_data)/2 < min_points_in_split or len(set(y_data)) == 1:
            self.is_leaf = True
            if tree_type.lower() == 'classification':
                if len(np.unique(y_data))==0:
                    self.node_value = 1
                else:
                    self.node_value = calculate_max_phat(y_data, np.unique(y_data))[0]

            else:
                self.node_value = y_data.mean()
        else:
            if tree_type == 'classification':
                was_reduction, min_feature_split, _, feature_dict = find_classification_split(x_data=x_data[self.random_features],
                                                                                              y_data=y_data,
                                                                                              min_points_in_split=min_points_in_split,
                                                                                              use_impurity=classification_impurity,
                                                                                              node_penalty=node_penalty)

            else:
                if is_boosted:
                    was_reduction, min_feature_split, _, feature_dict = find_regression_boosted_split(x_data=x_data[self.random_features],
                                                                                                      y_data=y_data,
                                                                                                      regularization_constant=regularization_constant,
                                                                                                      node_penalty=node_penalty,
                                                                                                      min_points_in_split=min_points_in_split)
                else:
                    was_reduction, min_feature_split, _, feature_dict = find_regression_split(x_data=x_data[self.random_features],
                                                                                              y_data=y_data,
                                                                                              node_penalty=node_penalty,
                                                                                              min_points_in_split=min_points_in_split)
            if was_reduction:
                self.split_on = min_feature_split
                self.split_at = feature_dict[min_feature_split]['split_value']
                x_data_0 = x_data[x_data[min_feature_split] <= self.split_at]
                y_data_0 = y_data[x_data[min_feature_split] <= self.split_at]
                x_data_1 = x_data[x_data[min_feature_split] > self.split_at]
                y_data_1 = y_data[x_data[min_feature_split] > self.split_at]
                self.new_node_1 = tree_node(stop_depth=self.stop_depth,
                                            current_depth=self.current_depth+1,
                                            x_data=x_data_0,
                                            y_data=y_data_0,
                                            min_points_in_split=self.min_points_in_split,
                                            tree_type=self.tree_type,
                                            classification_impurity=self.classification_impurity,
                                            name=self.name+'0',
                                            node_dict = self.node_dict)
                self.new_node_2 = tree_node(stop_depth=self.stop_depth,
                                            current_depth=self.current_depth+1,
                                            x_data=x_data_1,
                                            y_data=y_data_1,
                                            min_points_in_split=self.min_points_in_split,
                                            tree_type=self.tree_type,
                                            classification_impurity=self.classification_impurity,
                                            name=self.name+'1',
                                            node_dict=self.node_dict)
            else:
                self.is_leaf = True
                if tree_type.lower() == 'classification':
                    if len(set(y_data)) == 0:
                        self.node_value = 1
                    else:
                        self.node_value = calculate_max_phat(y_data, list(set(y_data)))[0]

                else:
                    self.node_value = y_data.mean()


    def evaluate_single(self,
                        single_x_data):
        """
        evaluates a single type of data.
        :param single_x_data: list,np.array, or pd.DataFrame
        :return: the classification or regression value
        """
        if self.was_trained:
            if self.is_leaf:
                return self.node_value
            else:
                if type(single_x_data) == np.ndarray or type(single_x_data) == list:
                    single_x_data = np.array(single_x_data)
                    if single_x_data.shape == (len(self.features),):
                        x_data = pd.DataFrame([single_x_data], columns = self.features)
                    elif single_x_data.shape == (1,len(self.features)):
                        x_data = pd.DataFrame(single_x_data, columns=self.features)
                    else:
                        raise TypeError('incorrect data shape')
                elif type(single_x_data) == type(pd.DataFrame()):
                    x_data = single_x_data
                else:
                    print('incorrect data type, use pandas DataFrame, numpy array, or list')
                    raise TypeError
                if (x_data[self.split_on] <= self.split_at).all():
                    return self.new_node_1.evaluate_single(x_data)
                else:
                    return self.new_node_2.evaluate_single(x_data)
        else:
            print('Tree must first be trained!')


    def get_many_results(self,
                         df: type(pd.DataFrame()),
                         results: list,
                         depth: int = 0,
                         )->None:
        """
        Shouldn't be called by the user. Use evaluate_many.
        """
        df = df.copy()
        if self.is_leaf:
            df['class'] = self.node_value
            results.append(df)
        else:
            self.new_node_1.get_many_results(df[df[self.split_on] <= self.split_at], results,  depth+1)
            self.new_node_2.get_many_results(df[df[self.split_on] > self.split_at], results,  depth+1)


    def evaluate_many(self,
                      df: type(pd.DataFrame())
                      )->np.ndarray:
        """
        Evaluates a data frame of examples
        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        a numpy array of regression or classification values
        """
        df = df.copy()
        results = []
        if self.was_trained:
            self.get_many_results(df, results,0)
            df_return = pd.concat(results).sort_index()
            to_return = df_return['class'].to_numpy()
            return to_return
        else:
            print('Tree must first be trained!')
            return np.array([])


def find_regression_split(x_data: type(pd.DataFrame()),
                          y_data: np.ndarray,
                          node_penalty: float = 0,
                          min_points_in_split: int = 1,
                          )->(bool, str, float, dict):
    """
    Determines the splitting feature and value for non boosted regression trees, shouldn't be called directly.
    Parameters
    ----------
    x_data: pd.DataFrame
    y_data: np.array
    node_penalty: float
    min_points_in_split: int

    Returns
    -------
    was_reduction: bool,
    min_feature_split: str,
    global_min_error: float,
    feature_dict: dict
    """
    features = x_data.columns
    feature_dict = {feature: {'error': np.inf,
                              'split_value': None,
                              'avg_less': None,
                              'avg_greater': None} for feature in features}
    old_error = ((y_data-y_data.mean())**2).sum()
    global_min_error = old_error-node_penalty
    min_feature_split = features[0]
    was_reduction = False
    for feature in features:
        feature_data = x_data[feature]
        feature_data = feature_data.to_numpy().copy()
        feat_y = y_data.copy()[feature_data.argsort()]
        feature_data.sort()
        avg_1 = {}
        avg_2 = {}
        min_error = old_error-node_penalty
        min_split = feature_data[0]
        num_duplicate = 1
        previous_value = None
        for index, value in enumerate(feature_data):
            if index+1 < min_points_in_split or len(feature_data)-index-1 < min_points_in_split:
                continue
            elif index != len(feature_data) - 1:
                if feature_data[index + 1] == value:
                    num_duplicate += 1

                else:
                    if len(avg_1) == 0:
                        avg_1[value] = feat_y[:index+1].mean()
                        avg_2[value] = feat_y[index+1:].mean()
                        num_in_avg1 = num_duplicate
                        num_in_avg2 = len(feature_data) - num_duplicate
                        previous_value = value
                    else:
                        avg_1[value] = (num_in_avg1 * avg_1[previous_value] + num_duplicate * feat_y[index]) / (num_in_avg1+num_duplicate)
                        avg_2[value] = (num_in_avg2 * avg_2[previous_value] - num_duplicate * feat_y[index]) / (num_in_avg2-num_duplicate)
                        num_in_avg1 += num_duplicate
                        num_in_avg2 -= num_duplicate
                    sum1 = ((feat_y[:index+1]-avg_1[value])**2).sum()
                    sum2 = ((feat_y[index+1:]-avg_2[value])**2).sum()
                    error = sum1+sum2
                    if error < min_error:
                        min_error = error
                        min_split = (value+feature_data[index+1])/2
                    num_duplicate = 1
        feature_dict[feature]['error'] = min_error
        feature_dict[feature]['split_value'] = min_split
        if min_error < global_min_error:
            was_reduction = True
            global_min_error = min_error
            min_feature_split = feature
    return was_reduction, min_feature_split, global_min_error, feature_dict



def find_regression_boosted_split(x_data: type(pd.DataFrame()),
                                  y_data: np.ndarray,
                                  regularization_constant: float = 0,
                                  node_penalty: float = 0,
                                  min_points_in_split: int = 1
                                  )->(bool, str, float, dict):
    """
    Determines the splitting feature and value for boosted regression trees, shouldn't be called directly.

    Parameters
    ----------
    x_data: pd.DataFrame
    y_data: np.array
    regularization_constant: float
    node_penalty: float
    min_points_in_split: int

    Returns
    -------
    was_reduction: bool,
    min_feature_split: str,
    global_min_error: float,
    feature_dict: dict
    """
    features = x_data.columns
    feature_dict = {feature: {'reduction': 0,
                              'split_value': None} for feature in features}
    global_max_reduction = 0
    min_feature_split = features[0]
    old_loss = (y_data.sum())**2/(len(y_data)+regularization_constant)
    was_reduction = False
    for feature in features:
        feature_data = x_data[feature]
        feature_data = feature_data.to_numpy().copy()
        feat_y = y_data.copy()[feature_data.argsort()]
        feature_data.sort()
        sum_1 = 0
        sum_2 = 0
        max_reduction = 0
        min_split = feature_data[0]
        for index, value in enumerate(feature_data):
            if index == 0 or index+1 < min_points_in_split or len(feature_data)-index-1 < min_points_in_split:
                continue
            elif index != len(feature_data) - 1:
                if index == 0:
                    sum_1 = feat_y[0]
                    sum_2 = feat_y[1:].sum()
                else:
                    sum_1 += feat_y[index]
                    sum_2 -= feat_y[index]
                if feature_data[index+1] != value:
                    left_loss = sum_1**2/(len(feat_y[:index+1])+regularization_constant)
                    right_loss = sum_2**2/(len(feat_y[index+1:])+regularization_constant)
                    reduction = .5*(left_loss+right_loss-old_loss)-node_penalty
                    if reduction > max_reduction:
                        max_reduction = reduction
                        min_split = (value+feature_data[index+1])/2
        feature_dict[feature]['reduction'] = max_reduction
        feature_dict[feature]['split_value'] = min_split
        if max_reduction > global_max_reduction:
            was_reduction = True
            global_max_reduction = max_reduction
            min_feature_split = feature
    return was_reduction, min_feature_split, global_max_reduction, feature_dict

def find_classification_split(x_data: type(pd.DataFrame()),
                              y_data: np.ndarray,
                              min_points_in_split: int = 1,
                              use_impurity: str ='gini_index',
                              node_penalty: float =0,
                              )->(bool, str, float, dict):
    """
    Determines the splitting feature and value for non classification trees, shouldn't be called directly.

    Parameters
    ----------
    x_data: pd.DataFrame
    y_data: np.array
    min_points_in_split: int
    use_impurity: str
    node_penalty: float

    Returns
    -------
    was_reduction: bool,
    min_feature_split: str,
    global_min_impurity: float,
    feature_dict: dict
    """
    if 'gini' in use_impurity.lower():
        impurity_func = calculate_gini_index
    elif 'entropy' in use_impurity.lower():
        impurity_func = calculate_cross_entropy
    elif 'misclassification' in use_impurity.lower():
        impurity_func = calculate_misclassification
    else:
        print('Impurity function unknown, using Gini Index')
        impurity_func = calculate_gini_index
    all_classes = list(set(y_data))
    old_impurity = impurity_func(y_data,all_classes)[0]
    was_reduction = False
    features = x_data.columns
    global_min_impurity = old_impurity-node_penalty
    min_feature_split = features[0]
    feature_dict = {feature: {'impurity': None,
                              'split_value': None,
                              'classification_1':None,
                              'classification_phat_1':None,
                              'classification_2':None,
                              'classification_phat_2':None
                              } for feature in features}

    for feature in features:
        feature_data = x_data[feature]
        feature_data = feature_data.to_numpy().copy()
        feat_y = y_data.copy()[feature_data.argsort()]
        classes = list(set(feat_y))
        feature_data.sort()
        min_impurity = old_impurity-node_penalty
        min_feature_value_split = feature_data[0]
        max_phat_y_value_1 = classes[0]
        max_phat_1 = 0
        max_phat_y_value_2 = classes[0]
        max_phat_2 = 0
        for index, value in enumerate(feature_data):
            if index+1 < min_points_in_split or len(feature_data)-index-1 < min_points_in_split:
                continue
            elif index != len(feature_data)-1:
                if feature_data[index+1] == value:
                    continue
                else:
                    impurity_1, y_class_1, phat_1 = impurity_func(feat_y_subset=feat_y[:index+1],
                                                                  y_classes=classes)
                    impurity_2, y_class_2, phat_2 = impurity_func(feat_y_subset=feat_y[index+1:],
                                                                  y_classes=classes)
                    impurity = impurity_2 + impurity_1
                    if impurity < min_impurity:
                        min_impurity = impurity
                        min_feature_value_split = (value+feature_data[index+1])/2
                        max_phat_y_value_2 = y_class_2
                        max_phat_2 = phat_2
        feature_dict[feature]['gini_index'] = min_impurity
        feature_dict[feature]['split_value'] = min_feature_value_split
        feature_dict[feature]['classification_2'] = max_phat_y_value_2
        feature_dict[feature]['classification_phat_2'] = max_phat_2
        if min_impurity < global_min_impurity:
            was_reduction = True
            global_min_impurity = min_impurity
            min_feature_split = feature
    return was_reduction, min_feature_split, global_min_impurity, feature_dict



def calculate_phat(feat_y_subset: np.ndarray,
                   y_class: list
                   )->float:
    """
    Calculates the ratio of a specific class of the training values in a classification tree.
    Shouldn't be called directly.

    Parameters
    ----------
    feat_y_subset: np.array
    y_class: int

    Returns
    -------
    phat: float
    """
    is_y_class = np.ones(feat_y_subset.shape)
    return is_y_class[feat_y_subset == y_class].sum()/len(feat_y_subset)

def calculate_max_phat(feat_y_subset: np.ndarray,
                       y_classes: list
                       )->(int, float):
    """
    Calculates the class with the largest ratio of training values in a classification tree.
    Shouldn't be called directly.

    Parameters
    ----------
    feat_y_subset: np.array
    y_class: int

    Returns
    -------
    phat: float
    """
    max_phat = 0
    max_phat_y_value = y_classes[0]
    for y_class in y_classes:
        phat = calculate_phat(feat_y_subset, y_class)
        if phat > max_phat:
            max_phat = phat
            max_phat_y_value = y_class
    return max_phat_y_value, max_phat

def calculate_misclassification(feat_y_subset: np.ndarray,
                                y_classes: list
                                )->(float,int,float):
    """
    Calculates the number of incorrectly classified training
    values if a node is set to a specific class.
    Shouldn't be called directly.
    To use this in tree creation set use_impurity to "misclassification"

    Parameters
    ----------
    feat_y_subset: np.array
    y_classes: list

    Returns
    -------
    misclassification_rate: float, best_classification: int, best_classification_ratio: float
    """
    max_phat = 0
    max_phat_y_value = y_classes[0]
    for y_class in y_classes:
        phat = calculate_phat(feat_y_subset, y_class)
        if phat > max_phat:
            max_phat = phat
            max_phat_y_value = y_class
    return 1-max_phat, max_phat_y_value, max_phat

def calculate_gini_index(feat_y_subset: np.ndarray,
                         y_classes: list
                         )->(float,int,float):
    """
    Calculates the gini index for a node.
    Shouldn't be called directly.
    To use this in tree creation set use_impurity to any string containing "gini"

    Parameters
    ----------
    feat_y_subset: np.array
    y_classes: list

    Returns
    -------
    gini_index: float, best_classification: int, best_classification_ratio: float
    """
    gini = 0
    max_phat = 0
    max_phat_y_value = y_classes[0]
    for y_class in y_classes:
        phat = calculate_phat(feat_y_subset, y_class)
        gini += phat*(1-phat)
        if phat > max_phat:
            max_phat = phat
            max_phat_y_value = y_class
    return gini, max_phat_y_value, max_phat

def calculate_cross_entropy(feat_y_subset: np.ndarray,
                            y_classes: list
                            )->(float, int, float):
    """
    calculates the cross entropy of a node.
    Shouldn't be called directly.
    To use this in tree creation set use_impurity to any string containing "entropy"

    Parameters
    ----------
    feat_y_subset: np.array
    y_classes: list

    Returns
    -------
    cross_entropy: float, best_classification: int, best_classification_ratio: float
    """
    entropy = 0
    max_phat = 0
    max_phat_y_value = y_classes[0]
    for y_class in y_classes:
        phat = calculate_phat(feat_y_subset, y_class)
        if phat != 0:
            entropy -= phat*np.log(phat)
        if phat > max_phat:
            max_phat = phat
            max_phat_y_value = y_class
    return entropy, max_phat_y_value, max_phat


def make_test_y_values1(x):
    if x < 10:
        return -1
    elif x < 20:
        return 1
    elif x < 40:
        return -1
    elif x < 50:
        return 1
    elif x < 60:
        return -1
    elif x < 80:
        return 1
    else:
        return -1

def make_test_y_values2(x):
    if x > .5:
        return 1
    else:
        return -1

def make_bootstrap_data(x_data: type(pd.DataFrame([])),
                        y_data: np.ndarray,
                        new_set_size: int=0
                        )->(type(pd.DataFrame([])), np.ndarray):
    if new_set_size <= 0:
        new_set_size = len(x_data)

    rand_indices = np.random.randint(0, len(x_data), new_set_size)
    new_x = x_data.loc[rand_indices]
    new_y = y_data[rand_indices]
    new_x = new_x.reset_index(drop=True)
    return new_x, new_y



if __name__ == '__main__':
    df1 = pd.DataFrame({'f1':[i for i in range(100)],
                        'f2':[0 for i in range(100)]})
    df2 = pd.DataFrame({'f1':[i for i in range(100)],
                        'f2':[1 for i in range(100)]})
    x_test = pd.concat([df1, df2]).reset_index(drop=True)
    y_test = x_test.apply(lambda row:make_test_y_values1(row.f1)*make_test_y_values2(row.f2), axis=1).to_numpy()

    x_data = pd.DataFrame({'f1':[np.random.randint(0,100) for i in range(100)],
                        'f2':[np.random.randint(0,2) for i in range(100)]})
    y_data = x_data.apply(lambda row:make_test_y_values1(row.f1)*make_test_y_values2(row.f2), axis=1).to_numpy()
    node = tree_node(stop_depth=20,
                     current_depth=0,
                     x_data=x_data,
                     y_data=y_data,
                     min_points_in_split=1,
                     tree_type='classification',
                     classification_impurity='gini_index',
                     use_random_features=False,
                     num_random_features=0)
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

