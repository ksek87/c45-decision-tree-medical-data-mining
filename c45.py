'''
    CSCI 4144 Data Mining and Warehousing - Project
    Title: Implementing C4.5 Decision Tree Algorithm for Medical Data Mining
    Author: Keelin Sekerka-Bajbus B00739421

    Program Description:

    Data Source:
    - UCI Machine Learning Repository, Thyroid Disease Data Set https://archive.ics.uci.edu/ml/datasets/thyroid+disease
        (Using allbp.data and allbp.test files, 2800 instances in training)

    References Consulted:
    [1] Data Mining (3rd Edition) Chapter 8 https://doi-org.ezproxy.library.dal.ca/10.1016/B978-0-12-381479-1.00008-3
    [2] Pandas library documentation https://pandas.pydata.org/docs/
    [3] https://stackoverflow.com/questions/32617811/imputation-of-missing-values-for-categories-in-pandas
    [4] collections Counter documentation https://docs.python.org/3/library/collections.html#collections.Counter
'''

import collections
import math
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle


class Node:
    def __init__(self, x, y, attribute_list, node_type):
        self.data = x
        self.labels = y
        self.attributes_list = attribute_list
        self.best_attribute = None
        self.split_criterion = None
        self.split_up_down = None
        self.node_type = node_type
        self.leaf_label = None
        self.depth = 0
        self.children = []
        self.parent = None

    def __lt__(self, other):
        return self.depth < other.depth

    def predict_leaf_class(self):
        """
            Computes the frequency of classes in partition D, output the leaf node label predicted class
        :return: pred_class
        """
        # takes frequency of classes in D to determine the majority class to set as output leaf label
        freq_classes = collections.Counter(self.labels)  # [4]
        pred_class = max(freq_classes, key=freq_classes.get)
        self.leaf_label = pred_class
        return pred_class

    def print_node(self):
        """
            Print node values
        """
        print('best att-', self.best_attribute, 'split_crit-', self.split_up_down, self.split_criterion, 'type-',
              self.node_type, 'depth-',
              self.depth, 'class label-', self.leaf_label)

    def copy(self):
        pass


class C45Tree:
    def __init__(self, attributes, data):
        self.tree_nodes = []
        self.depth = 0
        self.num_leaves = 0
        self.root_node = None
        self.attributes = attributes[:-1]
        self.dataset = data

    def train(self, x_train, y_train):
        """
            Helper function to grow tree recursively, creates root node for the tree and initializes the recursion for
            training the tree.
        :param x_train:
        :param y_train:
        """
        # create root node, put data partition in node
        self.root_node = Node(x_train, y_train, self.attributes, 'root')
        self.tree_nodes.append(self.root_node)
        # call grow_tree with root node as base
        self.grow_tree(self.root_node, self.attributes, (x_train, y_train))

    def grow_tree(self, prev_node, attribute_list, D):
        """
            Uses C4.5 decision tree algorithm to grow a tree during training, based on pseudocode from [1].
        :param attribute_list:
        :param D:
        :param prev_node:
        :return: N, the new node
        """
        if prev_node is not None and prev_node.parent is not None:
            if prev_node not in prev_node.parent.children:
                prev_node.parent.children.append(prev_node)

        dup_N_flag = 0
        # check for termination cases
        # check if all tuples in D are in the same class
        if self.check_same_class_labels(D[1]):
            N = Node(D[0], D[1], attribute_list, 'leaf')
            N.depth = prev_node.depth + 1
            N.predict_leaf_class()  # determine the class of the leaf
            N.best_attribute = str(prev_node.best_attribute)
            N.split_up_down = prev_node.split_up_down
            N.split_criterion = prev_node.split_criterion
            self.tree_nodes.append(N)
            prev_node.children.append(N)
            N.parent = prev_node
            return N

        # check if attribute list is empty, do majority voting on class
        if not attribute_list:
            N = Node(D[0], D[1], attribute_list, 'leaf')
            N.depth = prev_node.depth + 1
            N.predict_leaf_class()  # determine the class of the leaf
            N.best_attribute = str(prev_node.best_attribute)
            N.split_criterion = prev_node.split_criterion
            N.split_up_down = prev_node.split_up_down
            self.tree_nodes.append(N)
            prev_node.children.append(N)
            N.parent = prev_node
            return N

        # create new node
        N = Node(D[0], D[1], attribute_list, 'node')
        N.depth = prev_node.depth + 1
        N.parent = prev_node
        # conduct attribute selection method, label node with the criterion
        best_attribute, crit_split_val = self.attribute_selection_method(D, attribute_list)

        N.best_attribute = best_attribute  # label node with best attribute
        N.split_criterion = crit_split_val  # for discrete
        if best_attribute == '':
            # early stop
            N.best_attribute = str(best_attribute)
            N.split_up_down = None
            N.node_type = 'leaf'
            N.data = prev_node.data
            N.labels = prev_node.labels
            N.predict_leaf_class()
            self.tree_nodes.append(N)
            prev_node.children.append(N)
            return N

        # remove split attribute from attribute list
        if best_attribute in attribute_list:
            attribute_list.remove(best_attribute)

        # check if attribute is discrete NOTE THIS LINE NEEDS TO BE MODIFIED FOR DIFFERENT DATASET
        if len(self.dataset[
                   best_attribute].unique()) > 5:  # max 5 discrete categories in attributes from Thyroid set
            # continuous, divide up data at mid point of the values ai + ai1/2
            l_part, r_part, split_val = self.continuous_attribute_data_partition(D, best_attribute)
            N.split_criterion = split_val
            N.split_up_down = 'UP'
            l_child = self.grow_tree(N, attribute_list, l_part)  # upper -> att_val > split_val
            N_V = Node(D[0], D[1], attribute_list, 'node')
            N_V.depth = N.depth
            N_V.best_attribute = best_attribute
            N_V.split_criterion = split_val
            N_V.parent = prev_node
            N_V.split_up_down = 'DOWN'
            r_child = self.grow_tree(N_V, attribute_list, r_part)  # lower -> att_val <= split_val
            N.children.append(l_child)
            N_V.children.append(r_child)
            N.parent = prev_node
            self.tree_nodes.append(N)
            self.tree_nodes.append(N_V)
            prev_node.children.append(N)
            prev_node.children.append(N_V)
            return N
        else:
            # discrete, partition based on unique values of attribute to create nodes for recursion
            vals = self.dataset[best_attribute].unique()  # D[0][best_attribute].unique()
            for v in list(vals):
                data_part = self.partition_data(D, best_attribute, v)

                if not data_part:  # TOGGLED TO EMPTY CAUSES 2 LEAVES ONLY TO BE MADE ** check this
                    # majority class leaf node computed of D
                    L = Node(D[0], D[1], attribute_list, 'leaf')
                    L.depth = N.depth + 1
                    L.best_attribute = best_attribute
                    L.split_criterion = v
                    L.predict_leaf_class()  # determine the class of the leaf
                    self.tree_nodes.append(L)
                    N.children.append(L)
                    L.parent = N
                else:
                    # recursion
                    N_V = Node(D[0], D[1], attribute_list, 'node')
                    N_V.depth = N.depth
                    N_V.best_attribute = best_attribute
                    N_V.split_criterion = v
                    N_V.parent = prev_node
                    N_V.parent.children.append(N_V)
                    child = self.grow_tree(N_V, attribute_list, data_part)
                    # self.tree_nodes.append(child)
                    dup_N_flag = 1
                    # prev_node.children.append(child)
                    # N.children.append(child)
                    # prev_node.children.append(N_V)
                    # N.split_criterion = crit_split_val

        if dup_N_flag == 0:
            if N not in self.tree_nodes:
                self.tree_nodes.append(N)
            prev_node.children.append(N)
        return N

    def continuous_attribute_data_partition(self, D, attribute):
        """
            Creates data partitions (left and right) for continuous attributes, computing the mid point that
            enables the best information gain ratio to be calculated from the partition.
        :param D:
        :param attribute:
        :return: l_part, r_part, split_val
        """
        # sort the data, find the value that will gain the max info gain ratio
        data = D[0].sort_values(by=[attribute])
        split_val = 0
        best_igr = 0
        l_part = []
        r_part = []

        for i in range(0, len(data) - 1):
            mid_point = (float(data.iloc[i][attribute]) + float(data.iloc[i + 1][attribute])) / 2
            left_d = D[0].loc[pd.to_numeric(D[0][attribute]) > mid_point]
            left_idx = D[0].index[pd.to_numeric(D[0][attribute]) > mid_point]
            left_y = D[1].loc[left_idx]
            right_d = D[0].loc[pd.to_numeric(D[0][attribute]) <= mid_point]
            right_idx = D[0].index[pd.to_numeric(D[0][attribute]) <= mid_point]
            right_y = D[1].loc[right_idx]
            igr = self.compute_info_gain_ratio_continuous(D, left_y, right_y)

            if igr >= best_igr:
                best_igr = igr
                split_val = mid_point
                l_part = (left_d, left_y)
                r_part = (right_d, right_y)

        return l_part, r_part, split_val

    def compute_info_gain_ratio_continuous(self, D, left_y, right_y):
        """
            Computes the information gain ratio for a continuous attribute partition
        :return info_gain_ratio
        """
        l_y = left_y
        r_y = right_y

        dataset_entropy = self.data_entropy(D[1])
        l_part_entropy = self.data_entropy(l_y)
        l_p_j = float(len(l_y) / len(D))
        l_ent = l_p_j * l_part_entropy
        r_part_entropy = self.data_entropy(r_y)
        r_p_j = float(len(r_y) / len(D))
        r_ent = r_p_j * r_part_entropy

        split_info = - self.split_info(l_p_j) - self.split_info(r_p_j)
        att_ent = l_ent + r_ent

        if split_info == 0:  # prevent division by zero for ratio
            return 0
        else:
            info_gain = self.information_gain(dataset_entropy, att_ent)
            info_gain_ratio = self.information_gain_ratio(info_gain,
                                                          split_info)
        return info_gain_ratio

    @staticmethod
    def check_same_class_labels(labels):
        """
            Checks set of labels to ensure they are of the same class type
        :param labels:
        :return: bool
        """
        if len(set(labels)) == 1:
            return True
        else:
            return False

    def attribute_selection_method(self, D, attribute_list):
        """
            Attribute Selection Method for decision tree as discussed in [1] (Figure 8.3), selects attribute that
            provides the best information gain ratio as a result.
        :param D:
        :param attribute_list:
        :return: best_attribute
        """
        best_attribute = ''
        dataset_entropy = self.data_entropy(D[1])
        splitting_criterion = ""
        best_info_gain_ratio = 0.0
        split_val = ''

        for attribute in attribute_list:
            # a_idx = self.attributes.get(attribute) MIGHT NEED THIS
            v = D[0][attribute].unique()  # find v distinct values of attribute
            att_ent = 0.0
            split_info = 0.0
            curr_val = ''
            val_ent = 0.0
            for val in v:
                data_partition = self.partition_data(D, attribute, val)
                partition_labels = data_partition[1]
                part_entropy = self.data_entropy(partition_labels)
                p_j = float(len(data_partition[1]) / len(D[1]))
                att_ent = att_ent + (p_j * part_entropy)
                split_info = split_info - self.split_info(p_j)

                if part_entropy > val_ent:
                    val_ent = part_entropy
                    curr_val = val

            # Best Attribute checks
            if split_info == 0:  # prevent division by zero for ratio
                continue
            else:
                info_gain = self.information_gain(dataset_entropy, att_ent)
                info_gain_ratio = self.information_gain_ratio(info_gain,
                                                              split_info)  # calculate info gain ratio to select

            # compare the top performing attribute info gain ratio
            if info_gain_ratio > best_info_gain_ratio:
                best_info_gain_ratio = info_gain_ratio
                best_attribute = attribute
                split_val = curr_val
        return best_attribute, split_val

    def class_prob(self, feature_label, labels):
        """
            Computes class probabilities from labels
        :param feature_label:
        :param labels:
        :return: p
        """
        c = collections.Counter(labels)  # [4]
        p = c[feature_label] / len(labels)
        return float(p)

    def data_entropy(self, labels):
        """
            Computes the Entropy, or Info(D) [1]
        :param labels:
        :return: entropy
        """
        entropy = 0.0
        class_freq = collections.Counter(labels)  # [4]
        for l in class_freq.keys():
            p = float(class_freq[l] / len(labels))
            entropy = entropy - math.log(p, 2)
        return entropy

    def information_gain(self, dataset_entropy, attribute_entropy):
        """
            Computes information gain based on the data entropy and attribute entropy [1]
        :param dataset_entropy:
        :param attribute_entropy:
        :return: gain
        """
        gain = dataset_entropy - attribute_entropy
        return gain

    def split_info(self, p_j):
        """
            Computes the information split, used in gain ratio [1]
        :param p_j:
        :return: info_split
        """
        # error protection for zero case
        if p_j == 0:
            return 0

        info_split = (p_j * math.log(p_j, 2))
        return info_split

    def information_gain_ratio(self, gain, split_info):
        """
            Computes information gain ratio [1]
        :param gain:
        :param split_info:
        :return:
        """
        gain_ratio = float(gain / split_info)
        return gain_ratio

    def partition_data(self, D, attribute, val):
        """
            Partitions a dataset D based on the value of a specific attribute
        :param D:
        :param attribute:
        :param val:
        :return: part, part_y
        """
        part = D[0].loc[D[0][attribute] == val]
        part_idx = D[0].index[D[0][attribute] == val]
        part_y = D[1].loc[part_idx]
        return part, part_y

    def test_tree(self, test_sample, node):
        """
            Using recursion, we go through each node (from the root through to the children) to find a leaf label
            to classify the test sample as a prediction.
        :param test_sample:
        :param node:
        :return: node.leaf_label, or recursion
        """

        if node.node_type == 'leaf':
            return node.leaf_label
        else:
            for child in node.children:
                if (child.best_attribute is None or child.best_attribute == '') and child.node_type == 'leaf':
                    return self.test_tree(test_sample, child)

                if (child.best_attribute is None or child.best_attribute == '') and child.node_type == 'node':
                    pass
                else:
                    if child.split_criterion == test_sample[child.best_attribute]:
                        return self.test_tree(test_sample, child)
                    else:
                        if child.split_up_down == 'UP':
                            # check if att_val > split_criterion
                            if pd.to_numeric(test_sample[child.best_attribute]) > float(child.split_criterion):
                                return self.test_tree(test_sample, child)
                            else:
                                pass
                        elif child.split_up_down == 'DOWN':
                            if pd.to_numeric(test_sample[child.best_attribute]) <= float(child.split_criterion):
                                return self.test_tree(test_sample, child)
                            else:
                                pass

    def predict(self, test_x, test_y):  # TODO Add this functionality from the code in main routine
        # uses test set to predict class labels from the constructed tree
        preds = []
        true_pred = 0
        for i in range(len(test_x)):
            tester_instance = test_x.iloc[i]
            pred = self.test_tree(tester_instance, self.root_node)
            # print(str(i), 'pred', pred, 'label', y.iloc[i])
            if pred == test_y.iloc[i]:
                true_pred += 1
            preds.append(pred)

        return true_pred, preds

    def print_tree(self):
        nodes_created = sorted(self.tree_nodes)
        for n in nodes_created:
            n.print_node()
            for d in n.children:
                d.print_node()
            print()
        return


# Main experiment routine, read dataset, dropna values using pandas, split x and y matrices to pass in to tree
# make test and training splits THYROID dataset
# declare tree, initialize root node / start training and growing the tree
# print the tree and stats
# conduct testing with test set for predictions, analyze results (accuracy, recall, etc)

# DATA LOADING AND PRE-PROCESSING TEST AND TRAINING DATA
column_names = ['age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant',
                'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre',
                'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4',
                'T4U measured', 'T4U', 'FTI measured', 'FTI', 'TBG measured', 'TBG', 'referral source', 'Class'
                ]

train_data = pd.read_csv('allbp_data.csv',
                         sep=' ,', names=column_names, encoding='utf-8', engine='python')
test_data = pd.read_csv('allbp_test.csv',
                        sep=' ,', names=column_names, encoding='utf-8', engine='python')

train_data[['index_dup', 'age']] = train_data['age'].str.split(',', n=1, expand=True)
train_data = train_data.drop('index_dup', 1)
train_data = train_data.replace('?', pd.NA)
# replace ? with most common value
train_data = train_data.fillna(train_data.mode().iloc[0])  # [3]

test_data[['index_dup', 'age']] = test_data['age'].str.split(',', n=1, expand=True)
test_data = test_data.drop('index_dup', 1)
test_data = test_data.replace('?', pd.NA)
# replace ? with most common value
test_data = test_data.fillna(train_data.mode().iloc[0])  # [3]

np.random.seed(42)  # replicate results using random seed
train_data = sklearn.utils.shuffle(train_data)
test_data = sklearn.utils.shuffle(test_data)

x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

y_train = y_train.replace('negative.', 'negative')
y_train = y_train.replace('increased  binding  protein.', 'increased  binding  protein')
y_train = y_train.replace('decreased  binding  protein.', 'decreased  binding  protein')
y_test = y_test.replace('negative.', 'negative')
y_test = y_test.replace('increased  binding  protein.', 'increased  binding  protein')
y_test = y_test.replace('decreased  binding  protein.', 'decreased  binding  protein')

# 100 sample decision tree
system_test = C45Tree(column_names, train_data)

f_out = open('initial_testing_results.txt', 'w')

# small sample of data
f_out.write('First Test: 100 Samples training, 24 Test Samples\n')
x = x_train[:100]
y = y_train[:100]
testing_x = x_train[101:125]
testing_y = y_train[101:125]

print('system_test:')
system_test.train(x, y)
f_out.write('Number of Nodes for 100 sample tree:' + str(len(system_test.tree_nodes)) + '\n')
nodes_created = system_test.tree_nodes

leaf_count = 0
for n in nodes_created:
    # print(n.print_node())
    if n.node_type == 'leaf':
        leaf_count += 1
print('leaves', leaf_count)
print(len(system_test.tree_nodes))
f_out.write('Number of leaves:' + str(leaf_count) + '\n')
# tester_instance = x_train.iloc[0]
# pred = system_test.test_tree(tester_instance, system_test.root_node)

true_pred, preds = system_test.predict(x,y)
print('train accuracy:', true_pred / len(x))
f_out.write('Training Accuracy:' + str(true_pred / len(x)))
true_pred, preds = system_test.predict(testing_x, testing_y)
print('test accuracy:', true_pred / len(testing_x))
f_out.write('\tTest Accuracy:' + str(true_pred / len(testing_x)) + '\n')

f_out.write('\nFirst Test: 500 Samples training, 124 Test Samples\n')
print('500 sample tests:')
x_500 = x_train[:500]
y_500 = y_train[:500]
testing_x = x_train[501:625]
testing_y = y_train[501:625]
system_test500 = C45Tree(column_names, train_data)
system_test500.train(x_500, y_500)
print('system500 nodes:', len(system_test500.tree_nodes))

leaf_count = 0
for n in system_test500.tree_nodes:
    # print(n.print_node())
    if n.node_type == 'leaf':
        leaf_count += 1
print('leaves', leaf_count)



f_out.write('Number of nodes:' + str(len(system_test500.tree_nodes)) + '\n')
f_out.write('\t Number of leaves:' + str(leaf_count) + '\n')
true_pred, preds = system_test500.predict(x_500,y_500)
print('train accuracy:', true_pred / len(x_500))
f_out.write('Train Accuracy:' + str(true_pred / len(x_500)))
true_pred, preds = system_test500.predict(testing_x,testing_y)
print('test accuracy:', true_pred / len(testing_x))  # RANDOM SEED 42, train acc=0.956 , test acc= 0.9677 55 nodes
f_out.write('\t Test Accuracy:' + str(true_pred / len(testing_x)))
leaf_count = 0

print('FULL SET')
f_out.write('\nFull Training Data Decision Tree (2800 samples)\n')
true_pred = 0
full_system = C45Tree(column_names, train_data)
full_system.train(x_train, y_train)
print(len(full_system.tree_nodes))
leaf_count = 0
for n in set(sorted(full_system.tree_nodes)):
    #print(n.print_node())
    if n.node_type == 'leaf':
        leaf_count += 1

f_out.write('Number of nodes:' + str(len(full_system.tree_nodes)) + '\n')
f_out.write('Number of leaves:' + str(leaf_count))
true_pred, preds = full_system.predict(x_train,y_train)
print('Full set train accuracy:', true_pred / len(x_train))
f_out.write('\nFull set train accuracy:' + str(true_pred / len(x_train)))
true_pred, preds = full_system.predict(x_test,y_test)
f_out.write("\tFull set test accuracy:"+ str(true_pred / len(x_test)))

f_out.close()

print()
full_system.print_tree()

print('FURTHER FULL ALLPB DATASET EXPERIMENTS')
res_out = open('full_experiments_allpb.txt', 'w')
res_out.write('FULL EXPERIMENTATION WITH ALLPB DATASET\n')
res_out.write('Experiment #1 - random state = 24 \n')

np.random.seed(24)  # replicate results using random seed
train_data = sklearn.utils.shuffle(train_data, random_state=24)
test_data = sklearn.utils.shuffle(test_data, random_state=24)
x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

y_train = y_train.replace('negative.', 'negative')
y_train = y_train.replace('increased  binding  protein.', 'increased  binding  protein')
y_train = y_train.replace('decreased  binding  protein.', 'decreased  binding  protein')
y_test = y_test.replace('negative.', 'negative')
y_test = y_test.replace('increased  binding  protein.', 'increased  binding  protein')
y_test = y_test.replace('decreased  binding  protein.', 'decreased  binding  protein')

exp1C45 = C45Tree(column_names, train_data)
exp1C45.train(x_train, y_train)
true_pred, preds = exp1C45.predict(x_train,y_train)
print('Full set train accuracy:', true_pred / len(x_train))
res_out.write('Train Accuracy:'+str(true_pred / len(x_train)))
true_pred, preds = exp1C45.predict(x_test,y_test)
print('Full set test accuracy:', true_pred / len(x_test))
res_out.write('\tTest Accuracy:'+str(true_pred / len(x_test)))


res_out.write('\nExperiment #2 - random state = 55 \n')
np.random.seed(55)  # replicate results using random seed
train_data = sklearn.utils.shuffle(train_data, random_state=55)
test_data = sklearn.utils.shuffle(test_data, random_state=55)

x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

y_train = y_train.replace('negative.', 'negative')
y_train = y_train.replace('increased  binding  protein.', 'increased  binding  protein')
y_train = y_train.replace('decreased  binding  protein.', 'decreased  binding  protein')
y_test = y_test.replace('negative.', 'negative')
y_test = y_test.replace('increased  binding  protein.', 'increased  binding  protein')
y_test = y_test.replace('decreased  binding  protein.', 'decreased  binding  protein')


exp2C45 = C45Tree(column_names, train_data)
exp2C45.train(x_train, y_train)
true_pred, preds = exp2C45.predict(x_train,y_train)
print('Full set train accuracy:', true_pred / len(x_train))
res_out.write('Train Accuracy:'+str(true_pred / len(x_train)))
true_pred, preds = exp2C45.predict(x_test,y_test)
print('Full set test accuracy:', true_pred / len(x_test))
res_out.write('\tTest Accuracy:'+str(true_pred / len(x_test)))


res_out.write('\nExperiment #3 - random state = 75 \n')
np.random.seed(75)  # replicate results using random seed
train_data = sklearn.utils.shuffle(train_data, random_state=75)
test_data = sklearn.utils.shuffle(test_data, random_state=75)

x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

y_train = y_train.replace('negative.', 'negative')
y_train = y_train.replace('increased  binding  protein.', 'increased  binding  protein')
y_train = y_train.replace('decreased  binding  protein.', 'decreased  binding  protein')
y_test = y_test.replace('negative.', 'negative')
y_test = y_test.replace('increased  binding  protein.', 'increased  binding  protein')
y_test = y_test.replace('decreased  binding  protein.', 'decreased  binding  protein')


exp3C45 = C45Tree(column_names, train_data)
exp3C45.train(x_train, y_train)
true_pred, preds = exp3C45.predict(x_train,y_train)
print('Full set train accuracy:', true_pred / len(x_train))
res_out.write('Train Accuracy:'+str(true_pred / len(x_train)))
true_pred, preds = exp3C45.predict(x_test,y_test)
print('Full set test accuracy:', true_pred / len(x_test))
res_out.write('\tTest Accuracy:'+str(true_pred / len(x_test)))
res_out.close()
