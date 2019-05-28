import numpy
import argparse
import random
import csv

import pandas as pd

from sklearn import tree, ensemble
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold,SelectFromModel
from sklearn import svm, linear_model, neighbors
from sklearn import metrics


import numpy
from sklearn import model_selection 
from sklearn import preprocessing

import argparse
import random
import csv
import numbers
import gc

import pandas as pd
from sklearn.metrics.scorer import _check_multimetric_scoring


def createFeatureMat(pairs, classes, drug_df, disease_df, featureMatfile=None):
    totalNumFeatures=drug_df.shape[1] + disease_df.shape[1]-2
    drug_features = drug_df.columns.difference( ['Drug'] )
    disease_features = disease_df.columns.difference( ['Disease'])
    featureMatrix = numpy.empty((0,totalNumFeatures), int)
    for pair,cls in zip(pairs,classes):
        (dr,di)=pair
        values1 = drug_df.loc[drug_df['Drug'] == dr][drug_features].values
        values2 = disease_df.loc[disease_df['Disease']==di][disease_features].values
        featureArray =numpy.append(values1,values2 )
        featureMatrix=numpy.vstack([featureMatrix, featureArray])
    return featureMatrix

def multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))
    return scores


def getData(goldindfile, drugfeatfiles, diseasefeatfiles, selectedFeatures=None):
    """
    return drug, disease features and gold standard (ground truth) data as DataFrame
    """
    if selectedFeatures != None:
        selectedFeatures += ['Drug','Disease']

    #Use delimiter with txt files. Remove delimiter with csv files
    gold_df= pd.read_csv(goldindfile, delimiter='\t')

    drugs=gold_df.Drug.unique()
    diseases=gold_df.Disease.unique()

    for i,featureFilename in enumerate(drugfeatfiles):
        temp=pd.read_csv(featureFilename, delimiter='\t')
        if i != 0:
            drug_df=drug_df.merge(temp,on='Drug')
            #drug_df=drug_df.merge(temp,how='outer',on='Drug')
        else:
            drug_df =temp

    #drug_df.fillna(0,inplace=True)
    
    
    #If feature selection is used, then removes all non-selected features from drug_df
    if selectedFeatures != None\
    :
        drug_feature_names = drug_df.columns.intersection(selectedFeatures)
        drug_df=drug_df[drug_feature_names]

    for i,featureFilename in enumerate(diseasefeatfiles):
        temp=pd.read_csv(featureFilename, delimiter='\t')
        if i != 0:
            disease_df=disease_df.merge(temp,on='Disease')
        else:
            disease_df =temp
    
    #If feature selection is used, then removes all non-selected features from disease_df
    if selectedFeatures != None:
        disease_feature_names = disease_df.columns.intersection(selectedFeatures)
        disease_df=disease_df[disease_feature_names]

    print ("number of drugs ",len(drug_df))
    print ("number of diseases ",len( disease_df))
    commonDrugs=set(drug_df['Drug'].unique()).intersection(set(drugs))
    commonDiseases=set(disease_df['Disease'].unique()).intersection(set(diseases))

    gold_df=gold_df.loc[gold_df['Drug'].isin(commonDrugs) & gold_df['Disease'].isin(commonDiseases) ] 
    drug_df=drug_df.loc[drug_df['Drug'].isin(gold_df.Drug.unique())]
    disease_df=disease_df.loc[disease_df['Disease'].isin(gold_df.Disease.unique())]
    print ("#drugs in gold ",len( drugs))
    print ("#diseases in gold ",len( diseases))
    print ("Used indications ",len(gold_df))
       
    return gold_df, drug_df, disease_df


def get_classification_model(model_type, model_fun = None, n_seed = None):
    """
    model_type: custom | svm | logistic | knn | tree | rf | gbc
    model_fun: the function implementing classifier when the model_type is custom
    The allowed values for model_type are custom, svm, logistic, knn, tree, rf, gbc
    corresponding to custom model provided in model_fun by the user or the default 
    models in Scikit-learn for support vector machine, k-nearest-neighbor, 
    decision tree, random forest and gradient boosting classifiers, respectively. 
    Returns the classifier object that provides fit and predict_proba methods.
    """
    if model_type == "svm":
        clf = svm.SVC(kernel='linear', probability=True, C=1)
    elif model_type == "logistic":
        clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, random_state=n_seed) #, fit_intercept=True, intercept_scaling=1, class_weight=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    elif model_type == "knn":
        clf = neighbors.KNeighborsClassifier(n_neighbors=5) #weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    elif model_type == "tree":
        clf = tree.DecisionTreeClassifier(criterion='gini', random_state=n_seed) #splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, class_weight=None, presort=False)
    elif model_type == "rf":
        clf = ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', random_state=n_seed) #, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, verbose=0, warm_start=False, class_weight=None)
    elif model_type == "gbc":
        clf = ensemble.GradientBoostingClassifier(n_estimators= 100, max_depth= 5, random_state = n_seed, max_features=0.9)
        #clf = ensemble.GradientBoostingClassifier(n_estimators=100, loss='deviance', learning_rate=0.1, subsample=1.0, random_state=n_seed) #, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
    elif model_type == "custom":
        if fun is None:
            raise ValueError("Custom model requires fun argument to be defined!")
        clf = fun
    else:
        raise ValueError("Uknown model type: %s!" % model_type)
    return clf

def getAbsoluteTrain(pairs, classes, n_proportion):
    """
    return absolute training set using whole data 
    """
    classes = numpy.array(classes)
    pairs = numpy.array(pairs)
    idx_false_list = []
    idx_true_list = []
    for idx, (pair, class_) in enumerate(zip(pairs, classes)):
        if class_ == 0:
            idx_false_list.append(idx)
        else:
            idx_true_list.append(idx)

    if n_proportion >= 1:
            indicies_train_negative = random.sample(idx_false_list, n_proportion * len(idx_true_list))
    else:
            indicies_train_negative = idx_false_list
    indices_train = idx_true_list + indicies_train_negative
    return pairs,  classes, indices_train


def makePredictions(goldindfile, modeltype, unlabeled, outputfilename, drugfeatfiles, diseasefeatfiles, featureselectionfile, n_proportion, n_seed ):
    """
    genereates prediction (probablity score for indication) for unlabeled data based on given gold standard (selected) features
    goldindfile : gold standard drug-disease relation data (only positive associations)
    modeltype : classification model name, could be one one of these:  svm | logistic | knn | tree | rf | gbc
    unlabeled : file contaning drug-disease pairs to be labeled 
    outputfilename : file to ouput predection for input unlabaled pairs
    drugfeatfiles : list of files for drug features
    diseasefeatfiles : list of files for disease features
    featureselectionfile : selected features (could be drug or disease feature)
    n_proportion : proportion of negative samples compared to positve drug-diseaase relations
    n_seed : a seed to generate the same machine learning model 
    """
    
    
    #Use delimiter with txt files. Remove delimiter with csv files
    pred_df = pd.read_csv(unlabeled, delimiter=',')

    
    #No Feature selection
    selectedFeatures =None
    
    #With Feature selection
    if featureselectionfile != None:
        df_sf = pd.read_csv(featureselectionfile)
        selectedFeatures = list(df_sf['feature'])
    
    #Gets goldstandard data and binary feature matrix for both the indicated drug and disease files
    gold_df, drug_df, disease_df = getData(goldindfile, drugfeatfiles, diseasefeatfiles, selectedFeatures)
    
    features=[ fn[fn.index('-')+1:fn.index('.txt')] for fn in drugfeatfiles+diseasefeatfiles]

    
    drugDiseaseKnown = set([tuple(x) for x in  gold_df[['Drug','Disease']].values])

    commonDrugs=drug_df['Drug'].unique()
    commonDiseases=disease_df['Disease'].unique()
    pairs=[]
    classes=[]
    print ("commonDiseases",len(commonDiseases))
    print ("commonDrugs",len(commonDrugs))
    for dr in commonDrugs:
        for di in commonDiseases:
            if (dr,di)  in drugDiseaseKnown:
                cls=1
            else:
                cls=0
            pairs.append((dr,di))
            classes.append(cls)

    pairs,  classes, train_indicies  = getAbsoluteTrain(pairs, classes, n_proportion)
    pairs_train_df = pd.DataFrame( list(zip(pairs[train_indicies,0],pairs[train_indicies,1],classes[train_indicies])),columns=['Drug','Disease','Class'])
	
    train_df=pd.merge( pd.merge(drug_df,pairs_train_df, on='Drug'),disease_df,on='Disease')
	
    features= train_df.columns.difference(['Drug','Disease','Class'])
	
    print ("train #",len(train_df))
    X=train_df[features].values
    y=train_df['Class'].values.ravel()
    
    model_fun=None
    clf= get_classification_model(modeltype, model_fun, n_seed)
    clf.fit(X,y)
    
    
    pred_df=pd.merge( pd.merge(drug_df, pred_df, on='Drug'),disease_df,on='Disease')

    X_new = pred_df[features].values
    probs = clf.predict_proba(X_new)
    print ("#features",X.shape[1])
    print ("test samples #:",len(X_new))
    pred_df['Prob'] = probs[:,1]
    other_columns = pred_df.columns.difference(features)
    pred_df[other_columns].sort_values(by='Prob',ascending=False).to_csv(outputfilename, index=False)


if __name__ =="__main__":

    parser =argparse.ArgumentParser()
    parser.add_argument('-g', required=True, dest='goldindications', help='enter path to the file for the gold standard drug indications')
    parser.add_argument('-t', required=True, dest='test', help='enter path to the file you want to get predictions for')
    parser.add_argument('-m', required=True, dest='modeltype', help='enter classification model (eg. logistic | knn | tree | rf | gbc ')
    parser.add_argument('-o', required=True, dest='output', help='enter path to the output file for model')
    parser.add_argument('-p', required=True, dest='proportion', help='enter number of proportion')
    parser.add_argument('-s', required=True, dest='seed', help='enter seed number to generate the same training model')
    parser.add_argument('-fs', required=False, dest='featureselection', help='enter path to the file for selected features')
    parser.add_argument('-dr', required=True, dest='drugfeat', nargs='+', help='enter path to file for drug features ')
    parser.add_argument('-di', required=True, dest='diseasefeat', nargs='+', help='enter path to file for disease features ')

    args= parser.parse_args()

    goldindfile = args.goldindications
    model_type = args.modeltype
    predictions = args.test
    output_file_name = args.output
    drugfeatfiles = args.drugfeat
    diseasefeatfiles = args.diseasefeat
    featureselectionfile =args.featureselection
    n_proportion = int(args.proportion)
	#Get parameters
    n_seed = int(args.seed)
    
    makePredictions(goldindfile, model_type, predictions, output_file_name, drugfeatfiles, diseasefeatfiles, featureselectionfile, n_proportion, n_seed )
	




	
	 
