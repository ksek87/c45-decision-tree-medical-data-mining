'''
    CSCI 4144 Data Mining and Warehousing - Project
    Title: Implementing C4.5 Decision Tree Algorithm for Medical Data Mining
    Author: Keelin Sekerka-Bajbus B00739421

    Program Description:

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
        self.best_attribute_split = None
        self.node_type = node_type
        self.leaf_label = None
        self.depth = 0

    def predict_leaf_class(self):
        # takes frequency of classes in D to determine the majority class to set as output leaf label
        return


class C45Tree:
    def __init__(self):
        self.tree_nodes = []
        self.depth = 0
        self.num_leaves = 0
        return

    def grow_tree(self):
        # generates the decision tree recursively
        return

    def predict(self, test_data):
        # uses test set to predict class labels from the constructed tree
        return

    def attribute_selection_method(self, node):
        return

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
# conduct testing with test set for predictions, analyze resutls (accuracy, recall, etc)
