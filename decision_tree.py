# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn import tree
import pydotplus
from IPython.display import Image
import math


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    dictn = {}
    for i in range(0,len(x)):
        value = x[i]
        if value in dictn :
            dictn[x[i]].append(i)
        else :
            dictn[x[i]] = []
            dictn[x[i]].append(i)
    return dictn




def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    entropyY = 0
    for i in set(np.array(y)):
        prob = (np.array(y) == i).sum()/len(np.array(y))
        entropyY = entropyY + prob * math.log2(prob)
    return -entropyY


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    entropy_Y = entropy(y)
    entropy_YX = 0
    X = partition(x)
    for j in X:
        y_temp = []
        for i in X[j]:
            y_temp.append(y[i])
        entropy_YX = entropy_YX + (x.count(j) / len(x)) * entropy(y_temp)
    return entropy_Y - entropy_YX


def create_attribute_value_pairs(x):
    """
    To create attribute value pairs. For example, if x has 2 features x1 and x2 and x1 can take values 1,2,3 and x2 can
    take values 1,5 Then the attribute value pairs are
    [(x1,1),
     (x1,2),
     (x1,3),
     (x2,1),
     (X2,5)]

    Returns the list of attribute value pairs
   """

    attribute_value_pairs = []
    for i in range(0,x.shape[1]):
        for value in set(x[:,i]):
            attribute_value_pairs.append((i,value))
    return attribute_value_pairs

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.

    # Creating attribute value pairs
    if attribute_value_pairs == None or attribute_value_pairs.__len__ == 0 :
        attribute_value_pairs = create_attribute_value_pairs(x)

    # Base case for maximum depth and no more attributes left to split
    if depth == max_depth or len(attribute_value_pairs) == 0:
        freq = np.bincount(np.array(y))
        return np.argmax(freq)
    # Base case for entire dataset labels being pure
    elif all(t == y[0] for t in y):
        return y[0]
    else:
        # Calculating the infomration gain for attribute value pairs
        max_info_gain = 0
        for attribute_val in attribute_value_pairs:
            tmp = []
            i =attribute_val[0]
            for j in range(0, len(x)):
                value = x[j][i]
                if value == attribute_val[1]:
                    tmp.append(1)
                else:
                    tmp.append(0)
            info_gain = mutual_information(tmp,y)
            # Choosing the best attribute value pair based on information gain
            if info_gain >= max_info_gain:
                max_info_gain = info_gain
                best_split = attribute_val

        # Another base case where the data is same for all features
        if max_info_gain == 0:
            freq = np.bincount(np.array(y))
            return np.argmax(freq)

        # Storing the indices of the best split and its value
        value = best_split[1]
        i = best_split[0]
        tmp = []
        for j in range(0,len(x)):
            tmp.append(x[j][i])
        partition_X = partition(tmp)
        best_list = partition_X[value]

        x_true = []
        x_false = []
        y_true = []
        y_false = []

        for i in range(0,len(x)):
            tmp = np.asarray(x[i])
            if i in best_list:
                x_true.append(tmp)
                y_true.append(y[i])
            else:
                x_false.append(tmp)
                y_false.append(y[i])
        attribute_value_pairs_true = attribute_value_pairs.copy()
        attribute_value_pairs_false = attribute_value_pairs.copy()

        # Removing the best split from attribute value pairs to avoid using the same attribute again for splitting
        attribute_value_pairs_true.remove(best_split)
        attribute_value_pairs_false.remove(best_split)

        # Creating the decision tree
        tree = {(best_split[0],best_split[1],True): id3(x_true,y_true,attribute_value_pairs_true,depth+1,max_depth), (best_split[0],best_split[1],False): id3(x_false,y_false,attribute_value_pairs_false,depth+1,max_depth)}
        return tree


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    try:
        len(list(tree.keys()))
    except Exception:
        return tree
    cur = list(tree.keys())[0]
    if x[cur[0]] == cur[1]:
        return  predict_example(x,tree[(cur[0],cur[1],True)])
    else:
        return predict_example(x,tree[(cur[0],cur[1],False)])



def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    count = 0
    for i in range(0, len(y_true)):
        if y_true[i] != y_pred[i]:
            count = count+1
    return count/len(y_true)


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':

    # Part (b)
    for i in range(1,4):
        train_path = "./Monks_data/monks-" + str(i) + ".train"
        test_path = "./Monks_data/monks-" + str(i) + ".test"

        # Load the training data
        M = np.genfromtxt(train_path, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]

        # Load the test data
        M = np.genfromtxt(test_path, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        train_error = {}
        test_error = {}

        for d in range(1,11):
            # Learn a decision tree of depth 3
            decision_tree = id3(Xtrn, ytrn, max_depth=d)

            # Computing the training error
            train_pred = [predict_example(x,decision_tree) for x in Xtrn]
            trn_error = compute_error(ytrn, train_pred)

            # Computing the testing error
            test_pred = [predict_example(x,decision_tree) for x in Xtst]
            tst_error = compute_error(ytst, test_pred)

            train_error[d] = trn_error
            test_error[d] = tst_error

        # Plotting the testing and training error for all depths
        plt.figure()
        plt.plot(list(train_error.keys()),list(train_error.values()), marker='o', linewidth=3, markersize=12)
        plt.plot(list(test_error.keys()), list(test_error.values()), marker='o', linewidth=3, markersize=12)
        plt.xlabel('Depth',fontsize=16)
        plt.ylabel('Train/Test Error', fontsize=16)
        plt.xticks(list(train_error.keys()),fontsize=12)
        plt.legend(['Train Error', 'Test Error'],fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.title("MONKS DATASET: "+ str(i))
        plt.show()


    # Part (c)
    # Load the training data
    M = np.genfromtxt('./Monks_data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./Monks_data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    for i in range(1,6,2):
        decision_tree = id3(Xtrn, ytrn, max_depth=i)

        # Pretty print it to console
        pretty_print(decision_tree)

        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, './monks1_learned_tree_'+str(i))

        # Compute the test error
        y_pred = [predict_example(x, decision_tree) for x in Xtst]

        print("MONKS-1 DATASET: Confusion matrix (Learned Decision Tree) for depth ",i)
        conf_matrix = confusion_matrix(ytst,y_pred)
        print(pd.DataFrame(conf_matrix,columns=['Predicted Positive', 'Predicted Negative'],index=['True Postive','True Negative']))

    # Part (d)
    # Decision Tree Classifier for Monks Dataset
    M = np.genfromtxt('./Monks_data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./Monks_data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    for i in range(1,6,2):
        data_feature_names = list(['X' + str(i) for i in range(1,len(Xtrn[0])+1)])
        decision_tree = tree.DecisionTreeClassifier(criterion='entropy',max_depth=i)
        decision_tree.fit(Xtrn,ytrn)

        dot_data = tree.export_graphviz(decision_tree, out_file=None,feature_names= data_feature_names,filled=True,rounded=True,special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('monks_1_sciketlearn_'+str(i)+'.png')
        Image(filename='monks_1_sciketlearn_'+str(i)+'.png')
        y_pred = decision_tree.predict(Xtst)

        print("MONKS-1 DATASET: Confusion marix( Sciketlearn Decision Tree) for depth ",i)
        conf_matrix = confusion_matrix(ytst, y_pred)
        print(pd.DataFrame(conf_matrix, columns=['Predicted Positive', 'Predicted Negative'],index=['True Postive', 'True Negative']))

    # Part (e)
    # For new Dataset Occupancy dataset
    # Load the training data
    M = np.genfromtxt('./occupancy_data/occupancy_train.txt', missing_values=0, skip_header=1, delimiter=',', dtype=int)
    ytrn = M[:, 7]
    Xtrn = M[:, 2:6]

    #Load the testing data
    M = np.genfromtxt('./occupancy_data/occupancy_test1.txt', missing_values=0, skip_header=1, delimiter=',', dtype=int)
    ytst = M[:, 7]
    Xtst = M[:, 2:6]

    # Converting continuous attributes to categorical attributes for training data
    for i in range(0,Xtrn.shape[1]):
        sum = 0
        count = 0
        for j in range(0,Xtrn.shape[0]):
            sum = sum + Xtrn[j][i]
            count = count+1
        mean = sum/count
        for j in range(0,Xtrn.shape[0]):
            if Xtrn[j][i] <= mean:
                Xtrn[j][i] = 0
            else:
                Xtrn[j][i] = 1

    # Converting continuous attributes to categorical attributes for test data
    for i in range(0,Xtst.shape[1]):
        sum = 0
        count = 0
        for j in range(0,Xtst.shape[0]):
            sum = sum + Xtst[j][i]
            count = count+1
        mean = sum/count
        for j in range(0,Xtst.shape[0]):
            if Xtst[j][i] <= mean:
                Xtst[j][i] = 0
            else:
                Xtst[j][i] = 1

    # Part (c) for Occupancy dataset
    for i in range(1,6,2):
        decision_tree = id3(Xtrn, ytrn, max_depth=i)

        # Pretty print it to console
        pretty_print(decision_tree)

        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, './occupancy_learned_tree_'+str(i))

        # Compute the test error
        y_pred = [predict_example(x, decision_tree) for x in Xtst]

        print("OCCUPANCY DATASET: Confusion matrix (Learned Decision Tree) for depth ",i)
        conf_matrix = confusion_matrix(ytst,y_pred)
        print(pd.DataFrame(conf_matrix,columns=['Predicted Positive', 'Predicted Negative'],index=['True Postive','True Negative']))

    # Part (d) for Occupancy dataset
    for i in range(1,6,2):
        data_feature_names = list(['X' + str(i) for i in range(1,len(Xtrn[0])+1)])
        decision_tree = tree.DecisionTreeClassifier(criterion='entropy',max_depth=i)
        decision_tree.fit(Xtrn,ytrn)

        dot_data = tree.export_graphviz(decision_tree, out_file=None,feature_names= data_feature_names,filled=True,rounded=True,special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('occupancy_sciketlearn_'+str(i)+'.png')
        Image(filename='occupancy_sciketlearn_'+str(i)+'.png')
        y_pred = decision_tree.predict(Xtst)

        print("OCCUPANCY DATASET: Confusion marix( Sciketlearn Decision Tree) for depth ",i)
        conf_matrix = confusion_matrix(ytst, y_pred)
        print(pd.DataFrame(conf_matrix, columns=['Predicted Positive', 'Predicted Negative'],index=['True Postive', 'True Negative']))


    #print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
