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


class Node:
    def __init__(self, x, y, node_type):
        self.data = x
        self.labels = y
        self.attributes_list = list(self.data.columns)
        self.splitting_attribute = None
        self.node_type = node_type
        self.leaf_label = None
        self.depth = 0
        self.children = []

    def predict_leaf_class(self):
        # takes frequency of classes in D to determine the majority class to set as output leaf label
        return


class C45Tree:
    def __init__(self):
        self.tree_nodes = []
        self.depth = 0
        self.num_leaves = 0
        self.root_node = None

    def train(self, x_train, y_train):
        '''
            Helper function to grow tree recursively, creates root node for the tree and initializes the recursion for
            training the tree.
        :param x_train:
        :param y_train:
        '''
        # create root node, put data partition in node
        self.root_node = Node(x_train, y_train, 'root')
        # call grow_tree with root node as base
        self.grow_tree(self.root_node)

    def grow_tree(self, prev_node):
        '''
            Uses C4.5 decision tree algorithm to grow a tree during training, based on pseudocode from [1].
        :param node:
        :return: N, the new node
        '''
        attribute_list = prev_node.attributes_list
        D = (prev_node.data, prev_node.labels)

        # check for termination cases
        # check if all tuples in D are in the same class
        if self.check_same_class_labels(D[1]):
            N = Node(D[0], D[1], 'leaf')
            N.depth = prev_node.depth + 1
            N.predict_leaf_class()  # determine the class of the leaf
            return N

        # check if attribute list is empty, do majority voting on class
        if not attribute_list:
            N = Node(D[0], D[1], 'leaf')
            N.depth = prev_node.depth + 1
            N.predict_leaf_class()  # determine the class of the leaf
            return N

        # create new node
        N = Node(D[0], D[1], 'node')
        N.depth = prev_node.depth + 1
        # conduct attribute selection method, label node with the criterion
        splitting_attribute = self.attribute_selection_method()  # TODO implement this
        N.splitting_attribute = splitting_attribute  # label node with the splitting criterion

        # go through each attribute, recursion as needed, return node as needed


    @staticmethod
    def check_same_class_labels(self, labels):
        if len(set(labels)) == 1:
            return True
        else:
            return False

    def predict(self, test_data):
        # uses test set to predict class labels from the constructed tree
        return

    def attribute_selection_method(self, node):
        criterion = []
        return criterion

    def information_gain(self):
        return

    def information_gain_ratio(self):
        return

    def print_tree(self):
        return


# Main experiment routine, read dataset, dropna values using pandas, split x and y matrices to pass in to tree
# make test and training splits THYROID dataset
# declare tree, initialize root node / start training and growing the tree
# print the tree and stats
# conduct testing with test set for predictions, analyze results (accuracy, recall, etc)
'''column_names = ['age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant',
                'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre',
                'tumor', 'hypopituitary','psych', 'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4',
                'T4U measured', 'T4U', 'FTI measured', 'FTI', 'TBG measured', 'TBG', 'referral source', 'Class'
                ]
train_data = pd.read_csv('allbp_data.csv',
                         sep=',', names=column_names, encoding='utf-8' )
print(len(train_data))
print(train_data.head())
print(train_data.columns)
print(train_data.tail())
'''