import sklearn # make sure this is installed in your environment.
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import random
from scipy.stats import ttest_ind
from scipy import stats

"""
[30 points] Complete parse_file_into_matrix
"""
def parse_file_into_matrix(file_name):
    """
    :param file_name: The path to the bank-additional-full.csv file
    :return:
    X: A D X 20 matrix, where D is the number of 'instances' or records in the file (the number of columns may be more
    than 20 if you use some alternate coding scheme than the one I recommended in encode_record_ (e.g., one-hot encoding)
    y: A D X 1 matrix containing only 1's (for 'yes' in the output variable) or 0's (for 'no')
    """
    dictionary_list = []

    # Read in using pandas
    myd = pd.read_csv(file_name, sep=";", header='infer')
    variable_name = list(myd.columns.values)

    #myd = myd.replace("unknown", "null") # replace unknown with -1

    # Convert category variables  https://pbpython.com/categorical-encoding.htmls
    for i in range(len(myd.columns)):
        if myd.dtypes.values[i] == np.object:  # if string
            # Saving the dict for each categories
            value = np.unique(myd[variable_name[i]])
            map = dict(zip(value, np.arange(len(value))))
            dictionary_list.append(map)
            print()
            print(variable_name[i], map, sep = " >>> ")

            # Converting
            lb_make = LabelEncoder()
            myd[variable_name[i]] = lb_make.fit_transform(myd[variable_name[i]])

    X = myd[variable_name[0:len(variable_name)-1]]
    y = myd[variable_name[-1]]  # last one
    return (X,y,dictionary_list)

dataX, datay,l  = parse_file_into_matrix("bank-additional-full.csv") # function designs only for this dataset
#print(dataX.head(10),datay.head(10))

"""
We will define 'positive' instances as those with label 1, and negative instances as those with label 0.
Q1 [5 points]. Write some code to count the number of 1s and 0s in y. How many positive and negative instances each
are in your dataset?
ANS:  use sum() since they are all 1s and 0s
"""
print("-----------------")
print("\n\ntotal positive: ", np.sum(datay)) # original data set has 4640

def training_testing_split(X, y, training_ratio=0.8):
    #combining X and y
    dataset = X
    dataset['y'] = y #41188 rows, 21 cols

    # postive/negative instance
    pos = dataset[dataset['y'] == 1]
    neg = dataset[dataset['y'] == 0]

    sample = random.randint(1, 10000000000)
    random.seed(sample)


    train_set_pos = pos.sample(frac=training_ratio) #
    train_set_neg = neg.sample(frac=training_ratio) #

    test_set_pos = pos.drop(train_set_pos.index) #
    test_set_neg = neg.drop(train_set_neg.index) #

    # concast both pos and neg and shuffle
    train = pd. concat([train_set_pos,train_set_neg], ignore_index=True)
    train = train.sample(frac=1).reset_index(drop=True) # frac=1 means return all rows (in random order)

    test = pd. concat([test_set_pos,test_set_neg], ignore_index=True)
    test = test.sample(frac=1).reset_index(drop=True)

    return (train.iloc[:, 0:20], train.y, test.iloc[:, 0:20], test.y)

"""
Q2 (20 points). Run the code above with the X and y that you got from parse_file_into_matrix, with training ratios
of 0.8, 0.5, 0.3 and 0.1. For each of these four cases, what is the 'ratio' of positive instances in the training
dataset to the total number of instances in the training dataset? Verify that this same ratio is achieved in the test
dataset. Write additional code to run these verifications if necessary (5 points per case).
"""
print("-----------------")
for i in [0.8,0.5,0.3,0.1]:
    print("\nRatio is: ", i)
    trainX, trainY, testX, testY = training_testing_split(dataX, datay, training_ratio=i)
    print(np.mean(trainY))
    print(np.mean(testY))
'''
Ratio is:  0.8
0.11265553869499241
0.11264870114105366
Ratio is:  0.5
0.11265417111780131
0.11265417111780131
Ratio is:  0.3
0.11265781806409841
0.11265260821309656
Ratio is:  0.1
0.11264870114105366
0.11265477892578704
'''

def train_models(X_train,y_train,model='decision_tree'):
    """
    In the code below, I have trained a model specifically for decision tree. You must expand the code to accommodate
    the other two models. To learn more about sklearn's decision trees, see https://scikit-learn.org/stable/modules/tree.html
    :param X_train: self-explanatory
    :param y_train:
    :param model: we will allow three values for model namely 'decision_tree', 'naive_bayes' and 'linear_SGD_classifier'
    (Hint: you must change the loss function in https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    to squared_loss to correctly implement the linear classifier. For the naive bayes, the appropriate model to use
    is the Bernoulli naive Bayes.)
    :return: trained model
    """
    if model == 'decision_tree':
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
    elif model == "naive_bayes":
        clf = BernoulliNB()
        clf = clf.fit(X_train, y_train)
    elif model == "linear_SGD_classifier":
        clf = linear_model.SGDClassifier(loss = "squared_loss") #max_iter=1000, tol=1e-3,
        clf = clf.fit(X_train, y_train)
    elif model == "NN":  # Neural Networks
        clf = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=500)
        clf = clf.fit(X_train,y_train)
    else:
        print("Error: must specify a model")
        return None

    return clf
"""
Code for this function has already been written
"""
def evaluate_model(X_test, y_test, model):
    y_predict = model.predict(X_test)
    return sklearn.metrics.f1_score(y_test, y_predict)  # print(classification_report(testY,predictions))

#clf  = train_models(trainX,trainY,model='NN')
#predictions = clf.predict(testX)
#print(evaluate_model(testX,testY, clf))

def trials():
    """
    taking a csv data set, train and fit models
    :return: F1 scores with multiple experiments
    """
    all_tables = np.zeros((5, 10, 3))

    # read in
    dataX, datay, l = parse_file_into_matrix("bank-additional-full.csv")

    ratio_list = [0.1, 0.3, 0.5, 0.7,0.9]
    methods = ["decision_tree", "naive_bayes", "linear_SGD_classifier"]

    # different ratio for spliting
    for r in range(len(ratio_list)):
        # 10 times for every ratio
        for trails in range(10):
            trainX, trainY, testX, testY = training_testing_split(dataX, datay, training_ratio=ratio_list[r])
            # fit every models
            for mol in range(len(methods)):
                clf = train_models(trainX, trainY, model = methods[mol])
                f1 = evaluate_model(testX, testY, clf)
                all_tables[r][trails][mol] = evaluate_model(testX, testY, clf)
    return all_tables



# run master and print out each table
result = trials()
pvalue_list = []
tvalue_list = []
pvalue_list1 = []
tvalue_list1 = []
cl = 0.05
trails = 10
t = stats.t.ppf(1 - cl, trails-1)

for i in range(len(result)):
     table = pd.DataFrame(result[i], columns=["decision_tree", "naive_bayes", "linear_SGD_classifier"])
     print("---------\n")
     print(table)

     # saving as csv
     export_csv = table.to_csv('result'+str(i)+'.csv', header=True)

     # Testing via One sided T-test # degree = 10-1 = 9

     # Let mu_1 be the mean of Decision_tree F1 score
     # Let mu_2 be the mean of naive_bayes F1 score

     # H0:  mu_1 >= mu_2
     # H1:  mu_1 < mu_2

     ttest, pval = ttest_ind(table["decision_tree"],table["naive_bayes"])  # note: returned two-tailed p-value.
     pvalue_list.append(pval)
     tvalue_list.append(ttest)

     #print("p-value", pval)
     #print("t-value", ttest)

     # for extra creidt
     ttest, pval = ttest_ind(table["decision_tree"],table["linear_SGD_classifier"])
     pvalue_list1.append(pval)
     tvalue_list1.append(ttest)

print("T value list for decision_tree and naive_bayes ",tvalue_list)
print("T value list for decision_tree and linear_SGD_classifier ",tvalue_list1)
'''
Ratio = 0.1
   decision_tree  naive_bayes  linear_SGD_classifier
0       0.503924     0.257579               0.000000
1       0.493801     0.212433               0.202497
2       0.498064     0.238625               0.000000
3       0.480349     0.200245               0.202497
4       0.523793     0.192543               0.000000
5       0.501514     0.138239               0.201160
6       0.513830     0.253105               0.000000
7       0.475422     0.260125               0.000000
8       0.497877     0.271327               0.000000
9       0.494807     0.195216               0.042573
---------
Ratio = 0.3
   decision_tree  naive_bayes  linear_SGD_classifier
0       0.486139     0.239559               0.202494
1       0.502854     0.248762               0.202494
2       0.504988     0.244975               0.202494
3       0.501614     0.240025               0.000000
4       0.518078     0.243655               0.202494
5       0.512506     0.201453               0.202333
6       0.516090     0.263442               0.202494
7       0.520149     0.265792               0.202494
8       0.525957     0.264116               0.202494
9       0.514778     0.273691               0.000000
---------
Ratio = 0.5
   decision_tree  naive_bayes  linear_SGD_classifier
0       0.511154     0.209229               0.199440
1       0.511528     0.254211               0.000000
2       0.528811     0.255069               0.000000
3       0.516074     0.242289               0.000000
4       0.507916     0.247674               0.202496
5       0.515504     0.242492               0.202496
6       0.502221     0.226297               0.000000
7       0.515119     0.238365               0.202496
8       0.511487     0.239135               0.167275
9       0.521414     0.242451               0.000000
---------
Ratio = 0.7
   decision_tree  naive_bayes  linear_SGD_classifier
0       0.517049     0.254937               0.000000
1       0.504727     0.250181               0.000000
2       0.519843     0.233971               0.000000
3       0.517730     0.256225               0.202502
4       0.498018     0.256278               0.000000
5       0.518362     0.257970               0.001431
6       0.522051     0.258415               0.001436
7       0.528880     0.235778               0.202502
8       0.503396     0.245166               0.202502
9       0.498027     0.258900               0.000000
---------
Ratio = 0.9
   decision_tree  naive_bayes  linear_SGD_classifier
0       0.520990     0.253363               0.004292
1       0.533475     0.257361               0.029474
2       0.515152     0.250540               0.000000
3       0.480435     0.241417               0.441026
4       0.533623     0.265060               0.202487
5       0.529284     0.238512               0.000000
6       0.516129     0.226966               0.202487
7       0.519871     0.257384               0.000000
8       0.506835     0.228635               0.000000
9       0.549540     0.252212               0.202487

'''


#null hypothesis is that the decision tree is better than naive bayes.
#alternate hypothesis is that the naive bayes is better than decision tree
#For each of the five training percentages,
#using 95% as the confidence level, can you reject the null hypothesis? State your p-values here for all five training
#percentages. Which test did you use?

# Testing via One sided T-test # degree = 10-1 = 9

# Let mu_1 be the mean of Decision_tree F1 score
# Let mu_2 be the mean of naive_bayes F1 score

# H0:  mu_1 >= mu_2
# H1:  mu_1 < mu_2
print("\nComparing with P value")
# Note P value is two sided
one_tail_pvalue = []
for p in pvalue_list:
    p = 1 - p/2 # we know t >0
    one_tail_pvalue.append(p)
    if p < cl:
        print("Reject Null Hypothesis")
    else:
        print("Fail to Reject Null Hypothesis")
print(one_tail_pvalue)  #[0.9999999999999983, 1.0, 1.0, 1.0, 1.0]
# We fail to reject the null hypothesis with 95% confidence

#[5 points Extra credit] Using Bonferroni correction, can you determine if the decision tree is the best model overall (for this question,
#you have to consider the linear classifier also) for training percentage of 50 assuming a confidence level of 95%?

# RATIO 0.5

# Let mu_1 be the mean of Decision_tree F1 score
# Let mu_2 be the mean of naive_bayes F1 score
# Let mu_3 be the mean of linear classifier F1 score

# H0,1:  mu_1 >= mu_2
# H1,1:  mu_1 < mu_2
# H0,2:  mu_1 >= mu_3
# H1,2:  mu_1 >= mu_3

# Bonnferroni method -reject H0 if min_i pi <= a/n
n =2 # number of comparsion
for i in range(5):
    p1 = one_tail_pvalue[i]
    p2 = 1 - pvalue_list1[i]/2
    min_p = np.min([p1,p2])
    if min_p <= cl/n:
        print(">>>Reject Null Hypothesis")
    else:
        print("Fail to Reject Null Hypothesis")

#Fail to Reject Null Hypothesis with 95% confidence and sugguest that decision tree is the best model overall.