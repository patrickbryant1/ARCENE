#!/usr/bin/env python
"""A program that uses factor analysis and randomforest (scikitlearn) to characterize
MS spectra for cancer and healthy classes.
"""



#Modules
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sys
import pdb

#Read files
def file_reader(file_path):
    '''Input = file path (str)
       Output = numpy array of items in files
    '''
    
    data = []
    with open(file_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\n')
        for row in reader:
            for x in row:
                x=x.split(' ')
                example = []
                for item in x:
                    if item:
                        item = int(item) #convert to int
                        example.append(item)
                data.append(example)
        data = np.asarray(data)
    return data

   
#Lists with data lists
#train data
train_data = file_reader('/home/patrick/arcene/data/arcene_train.data')
#train labels
train_labels = file_reader('/home/patrick/arcene/data/arcene_train.labels')
train_labels = np.ravel(train_labels) #convert to 1d array
#validation data
valid_data = file_reader('/home/patrick/arcene/data/arcene_valid.data')
#validation labels
valid_labels = file_reader('/home/patrick/arcene/data/arcene_valid.labels')
valid_labels = np.ravel(valid_labels) #convert to 1d array



def parameter_optimization(param_grid, pipe):
    '''A function for optimizing parameters that uses scikit's GridSearchCV optimising on precision (TP/P) and recall (TP/TP+FN).
        Outputs a file for each optimization with grid scores and a detailed classification report. 
        Input = parameter grid (dict)
        Output = estimators with optimized parameter values (dict)
    '''

    #Scores to optimize for
    scores = ['precision', 'recall']
    best_classifiers = {}
    store_means = {}
    store_stds = {}

    for score in scores:
        with open(score + '.txt', 'w') as file:
  	    file.write("# Tuning hyper-parameters for %s" % score + '\n')

            #Grid search and cross validate
            #cv determines the cross-validation splitting strategy, 5 specifies the number of folds (iterations) in a (Stratified)KFold
            clf = GridSearchCV(pipe,
                               param_grid= param_grid, cv=5, 
                               scoring='%s_macro' % score)
            #Fit on train_data and write to file
            clf.fit(train_data,train_labels)
            #Write to file
            file.write("Best parameters set found on development set:" + '\n')
            file.write(str(clf.best_params_))
            file.write('\n' + '\n' + "Grid scores on development set:" + '\n')
            means = clf.cv_results_['mean_test_score']
            store_means[score] = means
            stds = clf.cv_results_['std_test_score']
            store_stds[score] = stds
            cv_results_params = clf.cv_results_['params']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                file.write('mean test score: ')
                file.write("%0.3f (+/-%0.03f) for %r"
                      % ( mean, std * 2, params) + '\n')
                
            
            #Predict on validation data and write to file
            file.write('\n' + "Detailed classification report:" + '\n')
            file.write("The model is trained on the full development set." + '\n')
            file.write("The scores are computed on the full evaluation set." + '\n')
            y_true, y_pred = valid_labels, clf.predict(valid_data)
            file.write(classification_report(y_true, y_pred))
            
            #Get the best classifier      
            best_classifiers[score] = clf.best_estimator_

    
    #Plot the training
    plot_training(store_means, store_stds, cv_results_params)   
        
    return best_classifiers

def plot_training(store_means, store_stds, cv_results_params):
    '''A function that plots the means and stds for the training using
    GridSearchCV and the values of the corresponding parameters.
    Input = store_means, store_stds, cv_results_params (dicts)
    '''
    
    organize_parameters = {'reduce_dim__n_components' : [], 'classify__n_estimators' : [],
    'classify__verbose' : [], 'classify__max_depth' : [], 'classify__min_samples_split' : []}
    number_of_combinations = []

    

    x= 0 #Number of combinations
    for params in cv_results_params:
        organize_parameters['reduce_dim__n_components'].append(params['reduce_dim__n_components'])
        organize_parameters['classify__n_estimators'].append(params['classify__n_estimators'])
	organize_parameters['classify__verbose'].append(params['classify__verbose'])
        organize_parameters['classify__max_depth'].append(params['classify__max_depth'])
        organize_parameters['classify__min_samples_split'].append(params['classify__min_samples_split'])
        x+=1
	number_of_combinations.append(x)

    #Plot grid and set y-axis
    plt.grid()
    axes = plt.gca()
    axes.set_ylim([-0.1,1.1])
    #Plot means and stds
    means_colors = {'precision': 'r', 'recall': 'g'} #colors
    for score in store_means:
        means = store_means[score]
        stds = store_stds[score]
        means_plus = [i + 2*j for i, j in zip(means, stds)]
        means_minus = [i - 2*j for i, j in zip(means, stds)]

        plt.fill_between(number_of_combinations, means_minus, means_plus, alpha=0.1,color=means_colors[score])
        plt.plot(number_of_combinations, means, 's-', color=means_colors[score], label= str(score))
   
   #Plot parameter values
    i = 0 #keep track of color
    clr_list = ['bo', 'ko', 'yo', 'co', 'mo'] #list of colors
    for key in organize_parameters:
        parameter = organize_parameters[key]
        label = str(parameter) + ' divided by ' + str(max(parameter)) #set label
        #Normalize
        parameter = [x /max(parameter) for x in parameter]
        plt.plot(number_of_combinations, parameter, clr_list[i], label = label)
        i += 1


    #plot x-label, set legend position and show plot
    plt.xlabel('combination')
    plt.legend(loc="best")   
    plt.show()

            
def plot_fa(fa):
    '''
    A function that plots the explained variance as a function
    of the number of training examples, in order to see how many
    are needed.
    input = factor analysis function
    '''
    fa.fit(train_data) #fit to train_data
    
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(fa.explained_variance_, linewidth=2) #plot explained variance
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.show()

###
#Main program
###

#Parameter grid
#Parameters of pipelines can be set using __ separated parameter names
param_grid = {'reduce_dim__n_components': [20,60],
        'classify__n_estimators': [10, 50],
	'classify__verbose': [0,10],
        'classify__max_depth': [5,15],
        'classify__min_samples_split': [2,10]}
   
#Create pipeline
random_forest = RandomForestClassifier()
fa = decomposition.FactorAnalysis()
pipe = Pipeline(steps=[('reduce_dim', fa), ('classify', random_forest)])
	

#Optimize parameters in parameter grid
best_classifiers = parameter_optimization(param_grid, pipe)

#Plot explained variance as a function of number of number of components
plot_fa(fa)
