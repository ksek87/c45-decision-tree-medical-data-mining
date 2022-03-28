'''
    CSCI 4144 Data Mining and Warehousing - Project
    Title: Implementing C4.5 Decision Tree Algorithm for Medical Data Mining
    Author: Keelin Sekerka-Bajbus B00739421

    Program Description:

    Data Source:
    - UCI Machine Learning Repository, Thyroid Disease Data Set https://archive.ics.uci.edu/ml/datasets/thyroid+disease
        Using thyroid0387.data file

    References Consulted:
    [1] Data Mining (3rd Edition) Chapter 8 https://doi-org.ezproxy.library.dal.ca/10.1016/B978-0-12-381479-1.00008-3
'''

import numpy as np
import pandas as pd
import itertools
import math
import csv
import collections


class Node:
    def __init__(self, x, y, attribute_list, node_type):
        self.data = x
        self.labels = y
        self.attributes_list = attribute_list
        self.best_attribute = None
        self.split_criterion = None
        self.node_type = node_type
        self.leaf_label = None
        self.depth = 0
        self.children = []
        self.parent = None

    def predict_leaf_class(self):
        # takes frequency of classes in D to determine the majority class to set as output leaf label
        freq_classes = collections.Counter(self.labels)
        pred_class = max(freq_classes, key=freq_classes.get)
        self.leaf_label = pred_class
        return pred_class


class C45Tree:
    def __init__(self, attributes, data):
        self.tree_nodes = []
        self.depth = 0
        self.num_leaves = 0
        self.root_node = None
        self.attributes = attributes[:-1]

    def train(self, x_train, y_train):
        '''
            Helper function to grow tree recursively, creates root node for the tree and initializes the recursion for
            training the tree.
        :param x_train:
        :param y_train:
        '''
        # create root node, put data partition in node
        self.root_node = Node(x_train, y_train, self.attributes, 'root')
        self.tree_nodes.append(self.root_node)
        # call grow_tree with root node as base
        self.grow_tree(self.root_node, self.attributes, (x_train, y_train))

    def grow_tree(self, prev_node, attribute_list, D):
        '''
            Uses C4.5 decision tree algorithm to grow a tree during training, based on pseudocode from [1].
        :param attribute_list:
        :param D:
        :param prev_node:
        :return: N, the new node
        '''
        # attribute_list = prev_node.attributes_list
        # D = (prev_node.data, prev_node.labels)

        # check for termination cases
        # check if all tuples in D are in the same class
        if self.check_same_class_labels(D[1]):
            N = Node(D[0], D[1], attribute_list, 'leaf')
            N.depth = prev_node.depth + 1
            N.predict_leaf_class()  # determine the class of the leaf
            self.tree_nodes.append(N)
            prev_node.children.append(N)
            N.parent = prev_node
            return N

        # check if attribute list is empty, do majority voting on class
        print(attribute_list, 'EMPTY CHECK')
        if not attribute_list:
            N = Node(D[0], D[1], attribute_list, 'leaf')
            N.depth = prev_node.depth + 1
            N.predict_leaf_class()  # determine the class of the leaf
            self.tree_nodes.append(N)
            prev_node.children.append(N)
            N.parent = prev_node
            return N

        # create new node
        N = Node(D[0], D[1], attribute_list, 'node')
        N.depth = prev_node.depth + 1
        # conduct attribute selection method, label node with the criterion
        best_attribute = self.attribute_selection_method(D, attribute_list)  # TODO implement this
        N.best_attribute = best_attribute  # label node with best attribute
        # remove split attribute from attribute list
        print(attribute_list)
        if best_attribute in attribute_list:
            attribute_list.remove(best_attribute)

        # check if attribute is discrete , TODO CHANGE THIS AFTER PREPROCESSING, DIFFERENT DATASET
        if len(D[0][best_attribute].unique()) > 5: # 5 referral sources.... TODO map the discrete and continuous columns
            # continuous, divy up data at mid point of the values ai + ai1/2
            l_part, r_part, split_val = self.continuous_attribute_data_partition(D, best_attribute)
            N.split_criterion = split_val
            l_child = self.grow_tree(N, attribute_list, l_part)
            r_child = self.grow_tree(N, attribute_list, r_part)
            self.tree_nodes.append(l_child)
            N.children.append(l_child)
            self.tree_nodes.append(r_child)
            N.children.append(r_child)
            N.parent = prev_node
        else:
            # discrete, partition based on unique values of attribute to create nodes for recursion
            vals = D[0][best_attribute].unique()

            for v in vals:
                data_part = self.partition_data(D, best_attribute, v)
                if not data_part:
                    # majority class leaf node computed
                    L = Node(data_part[0], data_part[1], 'leaf')
                    L.depth = prev_node.depth + 1
                    L.best_attribute = best_attribute
                    L.split_criterion = v
                    L.predict_leaf_class()  # determine the class of the leaf
                    self.tree_nodes.append(L)
                    prev_node.children.append(L)
                    L.parent = N
                else:
                    # recursion
                    child = self.grow_tree(N, attribute_list, data_part)
                    self.tree_nodes.append(child)
                    N.children.append(child)
                    N.parent = prev_node

        self.tree_nodes.append(N)
        prev_node.children.append(N)
        return N

    def continuous_attribute_data_partition(self, D, attribute):
        # sort the data, find the value that will gain the max info gain ratio
        data = D[0].sort_values(by=[attribute])
        split_val = 0
        best_igr = 0
        l_part = []
        r_part = []

        for i in range(0,len(data)-1):
            mid_point = int(data.iloc[i][attribute] + data.iloc[i + 1][attribute]) / 2
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

        l_y = left_y
        r_y = right_y

        dataset_entropy = self.data_entropy(D[1])
        l_part_entropy = self.data_entropy(l_y)
        l_p_j = float(len(l_y) / len(D))
        l_ent = l_p_j * l_part_entropy
        r_part_entropy = self.data_entropy(r_y)
        r_p_j = float(len(r_y) / len(D))
        r_ent = r_p_j * r_part_entropy
        print(l_p_j,r_p_j)

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
        if len(set(labels)) == 1:
            return True
        else:
            return False

    def predict(self, test_data):
        # uses test set to predict class labels from the constructed tree
        return

    def attribute_selection_method(self, D, attribute_list):
        best_attribute = attribute_list[0]
        dataset_entropy = self.data_entropy(D[1])
        splitting_criterion = ""
        best_info_gain_ratio = 0.0

        for attribute in attribute_list:
            # a_idx = self.attributes.get(attribute) MIGHT NEED THIS
            v = D[0][attribute].unique()  # find v distinct values of attribute
            att_ent = 0.0
            split_info = 0.0
            split_val = ''

            for val in v:
                data_partition = self.partition_data(D, attribute, val)
                partition_labels = data_partition[1]
                part_entropy = self.data_entropy(partition_labels)
                p_j = float(len(data_partition[1]) / len(D[1]))
                att_ent = att_ent + (p_j * part_entropy)
                split_info = split_info - self.split_info(p_j)

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

        print('best ATtribute', best_attribute)
        return best_attribute

    def class_prob(self, feature_label, labels):
        c = collections.Counter(labels)
        p = c[feature_label] / len(labels)
        return float(p)

    def data_entropy(self, labels):
        entropy = 0.0
        class_freq = collections.Counter(labels)
        for l in class_freq.keys():
            p = float(class_freq[l] / len(labels))
            entropy = entropy - math.log(p, 2)
        return entropy

    def information_gain(self, dataset_entropy, attribute_entropy):
        gain = dataset_entropy - attribute_entropy
        return gain

    def split_info(self, p_j):
        # error protection for zero case
        if p_j == 0:
            return 0

        info_split = (p_j * math.log(p_j, 2))
        print(info_split)
        return info_split

    def information_gain_ratio(self, gain, split_info):
        gain_ratio = float(gain / split_info)
        return gain_ratio

    def print_tree(self):
        return

    def partition_data(self, D, attribute, val):
        part = D[0].loc[D[0][attribute] == val]
        part_idx = D[0].index[D[0][attribute] == val]
        part_y = D[1].loc[part_idx]
        return part, part_y


# Main experiment routine, read dataset, dropna values using pandas, split x and y matrices to pass in to tree
# make test and training splits THYROID dataset
# declare tree, initialize root node / start training and growing the tree
# print the tree and stats
# conduct testing with test set for predictions, analyze results (accuracy, recall, etc)
column_names = ['age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant',
                'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre',
                'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4',
                'T4U measured', 'T4U', 'FTI measured', 'FTI', 'TBG measured', 'TBG', 'referral source', 'Class'
                ]

train_data = pd.read_csv('allbp_data.csv',
                         sep=' ,', names=column_names, encoding='utf-8', engine='python')

train_data[['index_dup', 'age']] = train_data['age'].str.split(',', n=1, expand=True)
train_data = train_data.drop('index_dup', 1)

# train_data = train_data.replace('?', pd.NA)
# CONSIDER CHANGING DATA TO ONLY NUMERIC TYPE?
print(len(train_data))
print(train_data.columns)
print(train_data['referral source'].unique())
x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
y_train = y_train.replace('negative.', 'negative')
y_train = y_train.replace('increased  binding  protein.', 'increased  binding  protein')
y_train = y_train.replace('decreased  binding  protein.', 'decreased  binding  protein')
print(y_train.head())
print()
# TESTS
node_test = Node(x_train, y_train, column_names[:-1], 'root')
print(node_test.__dict__)
print()
system_test = C45Tree(column_names, train_data)
print(system_test.__dict__)

print(system_test.check_same_class_labels(y_train))  # good
# feature attribute testing methods
# p_i calculation
print(set(y_train.values))
count_of_val = len(y_train[y_train == 'negative'])
print(count_of_val, 'expected prob', count_of_val / len(y_train))
p_i = system_test.class_prob('negative', y_train)
print(p_i)

# entropy (info gain) calcs
entr = system_test.data_entropy(y_train)
print('data entropy', entr)

# small sample of data
x = x_train[:100]
y = y_train[:100]

system_test.train(x, y)
print(system_test.tree_nodes)
