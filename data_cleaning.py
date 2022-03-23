"""
data_cleaning.py
desc:   Script to clean issues in UCI dataset for better processing in experiments, generate csv files for data
author: Keelin Sekerka-Bajbus

References:
     [1] https://stackoverflow.com/questions/21147058/pandas-to-csv-output-quoting-issue
"""

import pandas as pd
import csv


data = pd.read_csv('allbp.data',
                         sep='|', names=['', 'no'], encoding='utf-8' )
data = data.drop('no',axis=1)
data.to_csv('allbp_data.csv', header=False,quoting=csv.QUOTE_NONE, quotechar="",  escapechar=" ")

test = pd.read_csv('allbp.test',
                         sep='|', names=['', 'no'], encoding='utf-8' )
test = test.drop('no',axis=1)
test.to_csv('allbp_test.csv', header=False,quoting=csv.QUOTE_NONE, quotechar="",  escapechar=" ")
