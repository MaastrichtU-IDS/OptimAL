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

def runModel( pairs, classes,  drug_df, disease_df , cv, n_subset, n_proportion, n_fold, model_type, model_fun, features, disjoint_cv, n_seed, n_setsel, verbose=True, output_f=None):
    clf= get_classification_model(model_type, model_fun, n_seed)
    all_auc = []
    all_auprc = []
    all_fs = []
    le_drug = preprocessing.LabelEncoder()
    le_dis = preprocessing.LabelEncoder()
    le_drug.fit(pairs[:,0])
    le_dis.fit(pairs[:,1])
    
    results = pd.DataFrame()

    for i, (train, test) in enumerate(cv):
        file_name = None # for saving results
        pairs_train = pairs[train]
        classes_train = classes[train]
        pairs_test = pairs[test]
        classes_test = classes[test]
        
        pairs_train_df = pd.DataFrame( list(zip(pairs[train,0],pairs[train,1],classes[train])),columns=['Drug','Disease','Class'])
        train_df=pd.merge( pd.merge(drug_df,pairs_train_df, on='Drug'),disease_df,on='Disease')

        train_df['Drug']=le_drug.transform(train_df['Drug'])
        train_df['Disease']=le_dis.transform(train_df['Disease'])
        features_cols= train_df.columns.difference(['Drug','Disease','Class'])
        X=train_df[features_cols].values
        y=train_df['Class'].values.ravel()

        pairs_test_df = pd.DataFrame( list(zip(pairs[test,0],pairs[test,1],classes[test])),columns=['Drug','Disease','Class'])
        test_df=pd.merge( pd.merge(drug_df,pairs_test_df, on='Drug'),disease_df,on='Disease')

        test_df['Drug']=le_drug.transform(test_df['Drug'])
        test_df['Disease']=le_dis.transform(test_df['Disease'])
        features_cols= test_df.columns.difference(['Drug','Disease','Class'])
        X_new=test_df[features_cols].values
        y_new=test_df['Class'].values.ravel()
        
        clf.fit(X,y)

        scoring = ['precision', 'recall', 'accuracy', 'roc_auc', 'f1', 'average_precision']
        scorers, multimetric = metrics.scorer._check_multimetric_scoring(clf, scoring=scoring)
        #print(scorers)
        scores = multimetric_score(clf, X_new, y_new, scorers)
        #print ("Fold",scores, file=output_f)
        results = results.append(scores, ignore_index=True)  
        del X, y
        del X_new, y_new
        del train_df, pairs_train_df
        del test_df, pairs_test_df
        gc.collect()
    
    return results



def getData(goldindfile, drugfeatfiles, diseasefeatfiles, selectedFeatures=None):
    if selectedFeatures != None:
        selectedFeatures += ['Drug','Disease']

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

    if selectedFeatures != None:
        drug_feature_names = drug_df.columns.intersection(selectedFeatures)
        drug_df=drug_df[drug_feature_names]

    for i,featureFilename in enumerate(diseasefeatfiles):
        temp=pd.read_csv(featureFilename, delimiter='\t')
        if i != 0:
            disease_df=disease_df.merge(temp,on='Disease')
        else:
            disease_df =temp

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


def get_groups(idx_true_list, idx_false_list, n_subset, n_proportion=1, shuffle=False):
    """
    >>> a = get_groups([[13,2,1],[14,3,4],[15,5,6]], [[7,8],[9,10],[11,12]], 1, 1, True)
    """
    n = len(idx_true_list)
    if n_subset != -1:
        n_subset = n_subset / n 
    for i in range(n):
        if n_subset == -1: # use all data
            if n_proportion < 1:
                indices_test = idx_true_list[i] + idx_false_list[i]
            else:
                indices_test = idx_true_list[i] + random.sample( idx_false_list[i], n_proportion * len(idx_true_list[i]))
        else:
            if shuffle:
                indices_test = random.sample(idx_true_list[i], n_subset) + random.sample(idx_false_list[i], n_proportion * n_subset)
            else:
                indices_test = idx_true_list[i][:n_subset] + idx_false_list[i][:(n_proportion * n_subset)]
        indices_train = []
        for j in range(n):
            if i == j:
                continue
            if n_subset == -1: # use all data
                if n_proportion < 1:
                    indices_train += idx_true_list[j] + idx_false_list[j]
                else:
                    indices_train += idx_true_list[j] + random.sample( idx_false_list[j], n_proportion * len(idx_true_list[j]))
            else:
                if shuffle:
                    indices_train += random.sample(idx_true_list[j], n_subset) + random.sample(idx_false_list[j], n_proportion * n_subset)
                else:
                    indices_train += idx_true_list[j][:n_subset] + idx_false_list[j][:(n_proportion * n_subset)]
        yield indices_train, indices_test
 
def balance_data_and_get_cv(pairs, classes, n_fold, n_proportion, n_subset=-1, disjoint=False, n_seed = None):
    """
    pairs: all possible drug-disease pairs
    classes: labels of these drug-disease associations (1: known, 0: unknown)
    n_fold: number of cross-validation folds
    n_proportion: proportion of negative instances compared to positives (e.g.,
    2 means for each positive instance there are 2 negative instances)
    n_subset: if not -1, it uses a random subset of size n_subset of the positive instances
    (to reduce the computational time for large data sets)
    disjoint: whether the cross-validation folds contain overlapping drugs (True) 
    or not (False)
    This function returns (pairs, classes, cv) after balancing the data and
    creating the cross-validation folds. cv is the cross validation iterator containing 
    train and test splits defined by the indices corresponding to elements in the 
    pairs and classes lists.
    """
    classes = numpy.array(classes)
    pairs = numpy.array(pairs)
    idx_true_list = [ list() for i in range(n_fold) ]
    idx_false_list = [ list() for i in range(n_fold) ]
    if disjoint:
        i_random = random.randint(0,100) # for getting the shuffled drug names in the same fold below
        for idx, (pair, class_) in enumerate(zip(pairs, classes)):
            drug, disease = pair
            if disjoint == 1:
                i = sum([ord(c) + i_random for c in drug]) % n_fold
            else:
                i = sum([ord(c) + i_random for c in disease]) % n_fold
            if class_ == 0:
                idx_false_list[i].append(idx)
            else:
                idx_true_list[i].append(idx)
        #print "+/-:", map(len, idx_true_list), map(len, idx_false_list),n_fold,n_proportion, n_subset
        cv = get_groups(idx_true_list, idx_false_list, n_subset, n_proportion, shuffle=True)
    else:
        indices_true = numpy.where(classes == 1)[0]
        indices_false = numpy.where(classes == 0)[0]
        if n_subset == -1: # use all data
            n_subset = len(classes)
        indices_true = indices_true[:n_subset]
        numpy.random.shuffle(indices_false)
        if n_proportion < 1:
            indices = indices_false
        else:
            #indices = numpy.random.choice(indices_false,size=(n_proportion*indices_true.shape[0]))
            indices = indices_false[:(n_proportion*indices_true.shape[0])]
        #print "+/-:", len(indices_true), len(indices), len(indices_false)
        pairs = numpy.concatenate((pairs[indices_true], pairs[indices]), axis=0)
        classes = numpy.concatenate((classes[indices_true], classes[indices]), axis=0) 
        skf = model_selection.StratifiedKFold( n_splits=n_fold, shuffle=True, random_state=n_seed)
        cv= skf.split(pairs, classes)
    return pairs, classes, cv

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

if __name__ =="__main__":
    
    parser =argparse.ArgumentParser()
    parser.add_argument('-g', required=True, dest='goldindications', help='enter path to file for drug indication gold standard ')
    parser.add_argument('-m', required=True, dest='modelfile', help='enter path to file for trained sklearn classification model ')
    parser.add_argument('-disjoint', required=True, dest='disjoint', help='enter disjoint [0,1,2]')
    parser.add_argument('-o', required=True, dest='output', help='enter path to output file for model')
    parser.add_argument('-p', required=True, dest='proportion', help='enter number of proportion')
    parser.add_argument('-dr', required=True, dest='drugfeat', nargs='+', help='enter path to file for drug features ')
    parser.add_argument('-di', required=True, dest='diseasefeat', nargs='+', help='enter path to file for disease features ')
    parser.add_argument('-fs', required=False, dest='featureselection', help='enter path to the file for selected features')
    parser.add_argument('-nr', required=True, dest='nrun', help='enter number of runs')
    parser.add_argument('-nf', required=True, dest='nfold', help='enter number of folds for each run')

    args= parser.parse_args()

    goldindfile=args.goldindications
    model_type=args.modelfile
    disjoint=int(args.disjoint)
    output_file_name=args.output
    drugfeatfiles=args.drugfeat
    diseasefeatfiles=args.diseasefeat
    n_proportion = int(args.proportion)
    featureselectionfile =args.featureselection
    n_run = int(args.nrun)
    n_seed = 205
    n_fold = int (args.nfold)
    
    #Get parameters
    n_seed = 205
    #random.seed(n_seed) # for reproducibility
    n_subset =-1
    
    #output_file=open( output_file_name,'a')

    
    #No Feature selection
    selectedFeatures =None
    
    #With Feature selection
    if featureselectionfile != None:
        df_sf = pd.read_csv(featureselectionfile)
        selectedFeatures = list(df_sf['feature'])
        
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

    model_fun=None

    results_runs = pd.DataFrame()
    #output_file.write("n_fold\tn_proportion\tn_setsel\tmodel type\tfeatures\tdisjoint\tauc.mean\tauc.sd\tauprc.mean\tauprc.sd\tf-score.mean\tf-score.sd\n")
    for i in range(n_run):
        if n_seed is not None:
            n_seed += i
            random.seed(n_seed)
            numpy.random.seed(n_seed)
        pairs_, classes_, cv = balance_data_and_get_cv(pairs, classes, n_fold, n_proportion, n_subset, disjoint, n_seed )
        results = runModel( pairs_, classes_, drug_df, disease_df, cv, n_subset, n_proportion, n_fold, model_type, model_fun, features, disjoint, n_seed, 1, verbose=True)
        results_runs = results_runs.append(results.mean(), ignore_index=True)
        print ("===============================")
        print ('Average Fold Scores of Run',i,)
        print (results.mean())
    print ("===============================")
    print ("Average Scores of All Runs")
    print(results_runs.mean())
    results_runs.to_csv(output_file_name,index=False)
    