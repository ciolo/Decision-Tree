import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time


class Node:

    def __init__(self, parent, n_attributes, entropy=None, attribute=None, value=None, result=None):
        self.__parent = parent
        self.__children = []
        self.__n_attributes = n_attributes
        self.__entropy = entropy
        self.__attribute = attribute
        self.__value = value
        self.__leaf = result

    def __str__(self):
        desc = "Node split on " + str(self.__attribute) + " with entropy: " + str(self.__entropy)
        return desc

    def add_child(self, node):
        self.__children.append(node)

    def get_n_attributes(self):
        return self.__n_attributes

    def get_entropy(self):
        return self.__entropy

    def get_children(self):
        return self.__children

    def get_attribute(self):
        return self.__attribute

    def get_value(self):
        return self.__value

    def get_leaf(self):
        return self.__leaf

    def set_entropy(self, entropy):
        self.__entropy = entropy

    def set_attribute(self, attribute):
        self.__attribute = attribute

    def set_value(self, value):
        self.__value = value

    def set_leaf(self, result):
        self.__leaf = result


class DecisionTree:

    def __init__(self, X, y, features):
        self.__X = X
        self.__y = y
        self.__attributes_gains = dict((k, 0) for k in features)
        self.__attributes_set = set(features)
        self.__n_attributes = len(y)
        self.__initial_entropy = DecisionTree.entropy(y)
        self.__root = Node(None, len(y), self.__initial_entropy)

    @staticmethod
    def split_set(X, attribute, value):
        if X[attribute].dtypes == object:
            set1 = X[X[attribute] == value]
            set2 = X[X[attribute] != value]
        else:
            set1 = X[X[attribute] >= value]
            set2 = X[X[attribute] < value]
        return set1, set2

    def predict(self, observation):
        nodes = self.__root

        while True:
            if nodes.get_leaf() is not None:
                return nodes.get_leaf()
            else:
                v = observation[nodes.get_attribute()]
                children = nodes.get_children()
                if observation.dtypes == object:
                    if v == nodes.get_value():
                        nodes = children[0]
                    else:
                        nodes = children[1]
                else:
                    if v >= nodes.get_value():
                        nodes = children[0]
                    else:
                        nodes = children[1]

    def show(self):
        nodes = self.__root.get_children()

        for n in nodes:
            print(n)
            if n.get_leaf() is not None:
                print(n.get_leaf())
            nodes += [ch for ch in n.get_children()]

    def learn_tree(self, X, m, node=None):

        if node is None:
            node = self.__root

        current_entropy = node.get_entropy()

        split_attribute = "first"
        self.__attributes_gains[split_attribute] = - np.inf

        labels = X["class"]
        labels = labels.value_counts()
        counts = [e for e in labels]
        counts.remove(max(counts))
        err = sum(counts)

        if 0 < err < m:
            node.set_leaf(DecisionTree.count_labels(X["class"]))
            return

        for attr in self.__attributes_set:
            s = X[attr]
            values = dict(s.value_counts())
            for value in values.keys():
                (set1, set2) = DecisionTree.split_set(X, attr, value)
                p = float(len(set1)) / len(X)

                ent1, ent2 = DecisionTree.entropy(set1["class"]), DecisionTree.entropy(set2["class"])
                entropy_after_split = p * ent1 + (1 - p) * ent2

                information_gain = current_entropy - entropy_after_split
                if information_gain < 0:
                    information_gain = -information_gain

                if information_gain > self.__attributes_gains[split_attribute] and len(set1) > 0 and len(set2) > 0:
                    split_attribute, split_attribute_value = attr, value
                    self.__attributes_gains[split_attribute] = information_gain
                    best_set = (set1, set2)
                    entropy_set = (ent1, ent2)

        if self.__attributes_gains[split_attribute] > 0:
            node.set_entropy(current_entropy)
            node.set_attribute(split_attribute)
            node.set_value(split_attribute_value)
            t_node = Node(node, len(best_set[0]["class"]), entropy_set[0])
            f_node = Node(node, len(best_set[1]["class"]), entropy_set[1])

            node.add_child(t_node)
            node.add_child(f_node)
            self.learn_tree(best_set[0], m, t_node)
            self.learn_tree(best_set[1], m, f_node)
        else:
            node.set_leaf(DecisionTree.count_labels(X["class"]))

    @staticmethod
    def count_labels(target):
        results = dict(target.value_counts())

        return results

    @staticmethod
    def entropy(outcomes):
        s = outcomes.value_counts()
        ent = 0.0
        for e in s:
            ent += - (e/sum(s) * np.log2(e/sum(s)))
        return ent


def encode_target(df_tr, df_te, target_column):
    df_tr_mod = df_tr.copy()
    df_te_mod = df_te.copy()
    targets = df_tr[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_tr_mod[target_column].replace(map_to_int, inplace=True)
    df_te_mod[target_column].replace(map_to_int, inplace=True)
    return df_tr_mod, df_te_mod, map_to_int

df = pd.read_csv("your-dataset.csv")
df = shuffle(df)
train_set = df.sample(frac=0.70)
test_set = df.drop(train_set.index)
df_tr, df_te, targets = encode_target(train_set, test_set, "class")
print("* targets encoded", targets, sep="\n", end="\n\n")
train_labels = df_tr['class'].values.tolist()
test_labels = df_te['class'].values.tolist()

y = df_tr["class"]
X = df_tr
row, columns = X.shape
features = list(df_tr.columns[:columns-1])

scores = []
scores2 = []

# start = time.time()
for i in np.arange(0, 15, 1):
    print(i)

    dt = DecisionTree(X, y, features)
    dt.learn_tree(X, i)

    preds_train = []
    for index, row in df_tr.iterrows():
        preds_train.append(list(dt.predict(row).keys())[0])
    scores.append(accuracy_score(train_labels, preds_train))

    preds_test = []
    for index, row in df_te.iterrows():
        preds_test.append(list(dt.predict(row).keys())[0])
    scores2.append(accuracy_score(test_labels, preds_test))

"""end = time.time()
print(end - start)"""

print(scores)
print(scores2)
plt.plot(scores, label='train scores')
plt.plot(scores2, label='test scores')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()

dt.show()
