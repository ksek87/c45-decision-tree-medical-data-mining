# CSCI 4144 Data Mining and Warehousing Project - Implementing C4.5 Decision Tree Algorithm for Medical Data Mining 

## Program Description:
        This program implements Quinlan's C4.5 decision tree from scratch, and conducts experiments using the UCI
        ML repo Thyroid disease dataset for binding proteins (allbp). Specifically, this program uses two classes,
        Node and C45Tree to construct the decision tree using the Information gain ratio from information theory.

## Repository Directory:
               
    - /original_data                # directory stores allbp.data and allbp.test from [3]
    - /printed_trees                # directory stores text files containing the printed trees from experiments                    
    - README.md 
    - allbp_data.csv                # Training Data from [3]
    - allbp_test.csv                # Testing Data from  [3]
    - c45.py                        # Core program that implements C4.5 algorithm and conducts experiments
    - data_cleaning.py              # Simple script to clean original data from [3] and convert it to csv file for processing by pandas
    - full_experiments_allpb.txt    # contains experimental results for fully trained trees
    - initial_testing_results.txt   # contains experimental results for 100-sample, 500-sample, and fully trained trees    


## How to reproduce experimental results:
1. Navigate to the project repository
2. Run the c45.py program with the following command: `python3 c45.py`
3. View progress in terminal and final results in initial_testing_results.txt and full_experiments_allpb.txt.

Note: data_cleaning.py is only used to convert files to csv, files are already in csv format for c45.py file processing.

## Results:
- Fully trained tree: average training accuracy of 95.32% and testing accuracy of 96.91% over three trials
- 100-sample tree: 96.00% training accuracy, 95.83% testing accuracy
- 500-sample tree: 99.00% training accuracy, 97.58% testing accuracy


## Data Source:
    UCI Machine Learning Repository, Thyroid Disease Data Set https://archive.ics.uci.edu/ml/datasets/thyroid+disease
    (Using allbp.data and allbp.test files, 2800 instances in training, 972 testing)
    Citation: D. Dua and C. Graff, “UCI Machine Learning Repository”. University of California, Irvine, School of Information and Computer      Sciences, 2017. 


## Resources Consulted (In-Code):
    [1] Data Mining (3rd Edition) Chapter 8 https://doi-org.ezproxy.library.dal.ca/10.1016/B978-0-12-381479-1.00008-3
    [2] Pandas library documentation https://pandas.pydata.org/docs/
    [3] https://stackoverflow.com/questions/32617811/imputation-of-missing-values-for-categories-in-pandas
    [4] collections Counter documentation https://docs.python.org/3/library/collections.html#collections.Counter

## References

    [1] S. Aljawarneh, A. Anguera, J. W. Atwood, J. A. Lara, and D. Lizcano, “Particularities of data mining 
        in medicine: lessons learned from Patient Medical Time Series Data Analysis,” EURASIP Journal 
        on Wireless Communications and Networking, vol. 2019, no. 1, Nov. 2019. 
    [2] Shomona Gracia Jacob and R. Geetha Ramani. 2012. Mining of classification patterns in clinical 
        data through data mining algorithms. In Proceedings of the International Conference on Advances 
        in Computing, Communications and Informatics (ICACCI '12). Association for Computing 
        Machinery, New York, NY, USA, 997–1003. DOI:https://doi.org/10.1145/2345396.2345557
    [3] D. Dua and C. Graff, “UCI Machine Learning Repository”. University of California, Irvine, School 
        of Information and Computer Sciences, 2017.
    [4] J. R. Quinlan, C4.5: Programs for Machine Learning. San Mateo, California: Morgan Kaufmann, 
        1993.
    [5] J. Han, M. Kamber and J. Pei, “8 – Classification: Basic Concepts” in Data Mining: Concepts and 
        Techniques, 3rd ed. Morgan Kauffman, 2012, ch.8, pp.327-391. DOI: 
        https://doi.org/10.1016/B978-0- 12-381479-1.00008-3
    [6] S. Ruggieri, "Efficient C4.5 [classification algorithm]," in IEEE Transactions on Knowledge and 
        Data Engineering, vol. 14, no. 2, pp. 438-444, March-April 2002, doi: 10.1109/69.991727.
    [7] G. Van Rossum, The Python Library Reference,release 3.10.4. Python Software Foundation, 2022.
    [8] C. R. Harris et al., “Array programming with NumPy”, Nature, vol 585, no 7825, bll 357–362, Sep 
        2020.
    [9] J. Reback et al., pandas-dev/pandas: Pandas. Zenodo, 2020.
    
    [10] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python”, Journal of Machine Learning
         Research, vol 12, bll 2825–2830, 2011
