"""
Filename: data_cleaning.py
Program Description:   Script to clean issues in UCI dataset for better processing in experiments, generate csv files for data
Author: Keelin Sekerka-Bajbus, B00739421

References:
     [1] https://stackoverflow.com/questions/21147058/pandas-to-csv-output-quoting-issue
"""

import pandas as pd
import csv

# read in allbp.data and allbp.test
data = pd.read_csv('original_data/allbp.data',
                   sep='|', names=['', 'no'], encoding='utf-8')
data = data.drop('no',axis=1)
data.to_csv('allbp_data.csv', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ") #index=False)

test = pd.read_csv('original_data/allbp.test',
                   sep='|', names=['', 'no'], encoding='utf-8')
test = test.drop('no',axis=1)
test.to_csv('allbp_test.csv', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ") #index=False)
