from sklearn import tree, ensemble
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold,SelectFromModel
from sklearn import svm, linear_model, neighbors
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix
import numpy
import random

def get_groups(idx_true_list, idx_false_list, n_subset, n_proportion=1, shuffle=False):
    """
    >>> a = get_groups([[13,2,1],[14,3,4],[15,5,6]], [[7,8],[9,10],[11,12]], 1, 1, True)
    """
    n = len(idx_true_list)
    if n_subset != -1:
        n_subset = n_subset / n 
    for i in xrange(n):
        if n_subset == -1: # use all data
            if n_proportion < 1:
                print len(idx_true_list[i]), len(idx_false_list[i])
                indices_test = idx_true_list[i] + idx_false_list[i]
            else:
		indices_test = idx_true_list[i] + random.sample( idx_false_list[i], n_proportion * len(idx_true_list[i]))
        else:
            if shuffle:
                indices_test = random.sample(idx_true_list[i], n_subset) + random.sample(idx_false_list[i], n_proportion * n_subset)
            else:
                indices_test = idx_true_list[i][:n_subset] + idx_false_list[i][:(n_proportion * n_subset)]
        indices_train = []
        for j in xrange(n):
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
    idx_true_list = [ list() for i in xrange(n_fold) ]
    idx_false_list = [ list() for i in xrange(n_fold) ]
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
        cv = cross_validation.StratifiedKFold(classes, n_folds=n_fold, shuffle=True, random_state=n_seed)
    return pairs, classes, cv

def getData(goldindfile, drugfeatfiles, diseasefeatfiles):
    """
    goldindfile: gold standard drug-disease list, each row is a (drug,disease) pair : frist column drug name; seconde column is disease name
    drugfeatfiles: list of drug feature such files as  drug-chemical substructure, drug-target
    diseasefeatfiles: list of disease feature such files as  disease-phonetype, disease-shared entity
    Reads drug disease features and goldstandard and convert into feature vector 
    """
    drugDiseaseKnown = dict()

    with open( goldindfile ) as fileDrugIndKnown:
        # skip header
        header=next(fileDrugIndKnown)
        header = header.strip().split("\t")
        for line in fileDrugIndKnown:
            line = line.strip().replace('"','').split("\t")
            drug=line[0]
            disease=line[1]
	    if len(line) == 2:
                cls=1
            else:
                cls=int(line[2])
            drugDiseaseKnown[(drug,disease)]=cls
     
    drugFeatureNames ={}
    drugFeatures={}
    for i,featureFilename in enumerate(drugfeatfiles):
        featureFile =file(featureFilename)
        header =featureFile.next()
        header =header.strip().split('\t')
        drugFeatureNames[i]=header[1:]
        for line in featureFile:
            line = line.strip().split('\t')
            if not drugFeatures.has_key(line[0]):
                drugFeatures[line[0]]={}
            
            drugFeatures[line[0]][i]= [ int(e) for e in line[1:]]

                                       
    diseaseFeatureNames={}
    diseaseFeatures={}
    for i,featureFilename in enumerate(diseasefeatfiles):
        featureFile =file(featureFilename)
        header =featureFile.next()
        header =header.strip().split('\t')
        diseaseFeatureNames[i]=header[1:]

        for line in featureFile:
            line = line.strip().split('\t')
            if not diseaseFeatures.has_key(line[0]):
                diseaseFeatures[line[0]]={}

            diseaseFeatures[line[0]][i]= [ int(e) for e in line[1:]]
                                          
    return drugDiseaseKnown, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames     

def hasAtLeastOneFeature(featuremat, featureNames):
    for index in featureNames:
        if featuremat.has_key(index): 
            return True
    
    return False


def existAllFeatures(featuremat, featureNames):
    allFeatureExist=True
    for index in featureNames:
        if not featuremat.has_key(index): 
            allFeatureExist=False
    
    return allFeatureExist

def getPosNegPairs(drugDiseaseKnown, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames, negativeSelScheme):
    drugs= set([ dr  for dr,di in drugDiseaseKnown])
    diseases= set([ di  for dr,di in drugDiseaseKnown])

    #diseasesWithFeatures= set([ di for di in diseaseFeatures.keys() if existAllFeatures(diseaseFeatures[di],diseaseFeatureNames) ] )
    #drugsWithFeatures =  set([ dr for dr in drugFeatures.keys() if existAllFeatures(drugFeatures[dr],drugFeatureNames)])
    diseasesWithFeatures= set([ di for di in diseaseFeatures.keys() if hasAtLeastOneFeature(diseaseFeatures[di],diseaseFeatureNames) ] )
    drugsWithFeatures =  set([ dr for dr in drugFeatures.keys() if hasAtLeastOneFeature(drugFeatures[dr],drugFeatureNames)])
    #diseasesWithFeatures = diseaseFeatures.keys()
    #drugsWithFeatures =  drugFeatures.keys()
    
    commonDrugs= drugs.intersection( drugsWithFeatures )
    commonDiseases= diseases.intersection( diseasesWithFeatures )
    
    pairs=[]
    classes=[]
    for (dr,di) in drugDiseaseKnown:
        if dr in commonDrugs and di in commonDiseases:
            if  drugDiseaseKnown[(dr,di)] == 1:
                cls=1
            else:
                cls=0
            pairs.append((dr,di))
            classes.append(cls) 
                                
    return pairs,classes

def getAllPossiblePairs(drugDiseaseKnown, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames, negativeSelScheme):
    drugs= set([ dr  for dr,di in drugDiseaseKnown])
    diseases= set([ di  for dr,di in drugDiseaseKnown])
    
    diseasesWithFeatures= set([ di for di in diseaseFeatures.keys() if existAllFeatures(diseaseFeatures[di],diseaseFeatureNames) ] )
    drugsWithFeatures =  set([ dr for dr in drugFeatures.keys() if existAllFeatures(drugFeatures[dr],drugFeatureNames)])
    #diseasesWithFeatures= set([ di for di in diseaseFeatures.keys() if hasAtLeastOneFeature(diseaseFeatures[di],diseaseFeatureNames) ] )
    #drugsWithFeatures =  set([ dr for dr in drugFeatures.keys() if hasAtLeastOneFeature(drugFeatures[dr],drugFeatureNames)])
    #diseasesWithFeatures = diseaseFeatures.keys()
    #drugsWithFeatures = drugFeatures.keys()
    commonDrugs= drugs.intersection( drugsWithFeatures )
    commonDiseases= diseases.intersection( diseasesWithFeatures )
    abridged_drug_disease = [(dr,di)  for  (dr,di)  in drugDiseaseKnown if dr in drugsWithFeatures and di in diseasesWithFeatures ]

    commonDrugs = set( [ dr  for dr,di in  abridged_drug_disease])
    commonDiseases  =set([ di  for dr,di in  abridged_drug_disease])
  
    
    print "Gold standard, associations: %d drugs: %d diseases: %d"%(len(drugDiseaseKnown),len(drugs),len(diseases))    
    print "Drugs with features: %d Diseases with features: %d"%(len(drugsWithFeatures),len(diseasesWithFeatures))
    print "commonDrugs: %d commonDiseases : %d"%(len(commonDrugs),len(commonDiseases))
    
    
    pairs=[]
    classes=[]
    nPos=0
    nNeg=0
    if negativeSelScheme == 1:
        for dr in commonDrugs:
            for di in commonDiseases:
                if (dr,di)  in drugDiseaseKnown:
                    cls=1
		    nPos+=1
                else:
                    cls=0
		    nNeg+=1
                pairs.append((dr,di))
                classes.append(cls) 
                    
    elif negativeSelScheme == 2:        
        #diseasesforneg=set(diseasesWithFeatures).difference(commonDiseases) # select diseases that no drugs was indicated for
        for dr in commonDrugs:
            for di in diseasesWithFeatures:
                if (dr,di) in drugDiseaseKnown:
                    cls=1
                else:
                    cls=0    
                pairs.append((dr,di))
                classes.append(cls)
    elif negativeSelScheme == 3:
        unknown = []
        for dr in commonDrugs:
            for di in commonDiseases:
                if (dr,di)  in drugDiseaseKnown:
                    cls=drugDiseaseKnown[(dr,di)]
                    pairs.append((dr,di))
                    classes.append(cls) 
                    if cls ==1:
                        nPos +=1
                    else:
                        nNeg+=1
                else:
                    unknown.append((dr,di))
        
        for pair in random.sample(unknown, nPos-nNeg ):
            pairs.append(pair)
            classes.append(0)     	
    
    print "# known associations ",nPos        
    print "# all pairs ",len(pairs)        
    return pairs,classes


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

def mergeRow(row, featuremat, featureNames):

    for index in featureNames:
        #print feature
        if featuremat.has_key(index):
            row =numpy.append(row, featuremat[index])
        else:
            row =numpy.append(row, numpy.zeros(len(featureNames[index])))
    return row

def createFeatureArray(pair, cls, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames, featureMatfile=None):

    (dr,di)=pair
    sep='\t'
    
    rowStr = dr+sep+di
    #print rowStr
    row = numpy.array([])
    row = mergeRow(row, drugFeatures[dr], drugFeatureNames)
            
    row = mergeRow(row, diseaseFeatures[di], diseaseFeatureNames)

    rowStr += sep+str(cls)
    if featureMatfile !=None:
        featureMatfile.write( rowStr+'\n' )
    return row

def createFeatureMat(pairs, classes, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames, featureMatfile=None):
    totalNumFeatures=0
    for i in drugFeatureNames:
        totalNumFeatures+=len(drugFeatureNames[i])
    for i in diseaseFeatureNames:
        totalNumFeatures+=len(diseaseFeatureNames[i])
        
    #print totalNumFeatures
    featureMatrix= numpy.empty((0,totalNumFeatures), int)
    for pair,cls in zip(pairs,classes):
        #print pair,cls
        featureArray= createFeatureArray(pair, cls, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames, featureMatfile=None)
        #print len(featureArray)
        #print len(featureArray),totalNumFeatures
        featureMatrix=numpy.vstack([featureMatrix, featureArray])
        #featureMatrix = numpy.append(featureMatrix, numpy.array(featureArray), axis=0)
    return featureMatrix

def get_absolute_train(pairs, classes, n_proportion):
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
    return indices_train

def get_test_for_drug(pairs, classes, drug, n_fold, n_proportion, n_subset=-1, disjoint=False, n_seed = None):
    
    test_indicies=[]
    for i,pair in enumerate(pairs):
    	c,p=pair
   	#if c == 'DB00234': # Reboxetine
   	if c == drug: # Reboxetine
        	test_indicies.append(i)

    classes = numpy.array(classes)
    pairs = numpy.array(pairs)
    pairs_test=pairs[test_indicies]
    classes_test=classes[test_indicies]
    

        
def get_absolute_test(pairs, classes, n_fold, n_proportion, n_subset=-1, disjoint=False, n_seed = None):
    #classes = numpy.array(classes)
    #pairs = numpy.array(pairs)
    idx_false_list = []
    idx_true_list = []
    for idx, (pair, class_) in enumerate(zip(pairs, classes)):
        if class_ == 0:
            idx_false_list.append(idx)
        else:
            idx_true_list.append(idx)

    return idx_false_list

def saveSelectedFeatures(drugFeatureNames, diseaseFeatureNames, selectedFeatures):
    index=0
    featureNames=[]
    all_features = []
    for fn in drugFeatureNames:
	all_features +=drugFeatureNames[fn]

    for fn in diseaseFeatureNames:
	all_features +=diseaseFeatureNames[fn]

    for index,feat in enumerate(all_features):
	if index in selectedFeatures:
		featureNames.append(feat)	

    with open("../data/selectedFeatures1.txt",'w') as featureFile:
	for s in featureNames:
		featureFile.write(s+"\n")	 
            

def trainModel( pairs, classes, train, drugFeatures, diseaseFeatures,  drugFeatureNames, diseaseFeatureNames, model_type, model_fun, n_seed):
    clf= get_classification_model(model_type, model_fun, n_seed)
    classes = numpy.array(classes)
    pairs = numpy.array(pairs)
    pairs_train = pairs[train]
    classes_train = classes[train]
    X_train = createFeatureMat(pairs_train, classes_train, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames,  featureMatfile=None)
    #sel = VarianceThreshold()
    #print X_train.shape
    #X_train=sel.fit_transform(X_train)
    #print X_train.shape
    randomlr = linear_model.RandomizedLogisticRegression( C=1, random_state=n_seed, selection_threshold=0.1)
    #sfm = SelectFromModel(clf)
    randomlr.fit(X_train,classes_train)
    X_train = randomlr.transform(X_train)
    print "number of seleceted features",X_train.shape[1]   
    joblib.dump(randomlr,"../data/models/randomlr.pkl") 
    selectedFeatures=randomlr.get_support(indices=True)
    print selectedFeatures
    saveSelectedFeatures(drugFeatureNames, diseaseFeatureNames, selectedFeatures)

    y_train = numpy.array(classes_train)
    clf.fit(X_train, y_train)
    return clf


def load_learn_model(model_load_file):
    from sklearn.externals import joblib
    clf = joblib.load(model_load_file)
    return clf

def make_predictions_for_drug( pairs, classes, drug, drugFeatures, diseaseFeatures,  drugFeatureNames, diseaseFeatureNames, n_subset, model_load_file, output_f=None):
    clf = load_learn_model(model_load_file) 
    test_indicies=[]
    for i,pair in enumerate(pairs):
        c,p=pair
        #if c == 'DB00234': # Reboxetine
        if c == drug and classes[i] != 1: # Reboxetine
                test_indicies.append(i)

    classes = numpy.array(classes)
    pairs = numpy.array(pairs)
    pairs_test=pairs[test_indicies]
    classes_test=classes[test_indicies]
    
    X = createFeatureMat(pairs_test, classes_test, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames,  featureMatfile=None)
    probs= clf.predict_proba(X) 

    scores = zip(pairs_test[:,0],pairs_test[:,1],probs[:,1])
    scores.sort(key=lambda tup: tup[2],reverse=True)
    for (drug,disease,prob) in scores:
	output_f.write( str(drug)+'\t'+str(disease)+'\t'+str(prob)+'\n')
   

def runModel( pairs, classes,drugFeatures, diseaseFeatures,  drugFeatureNames, diseaseFeatureNames, cv, n_subset, n_proportion, n_fold, model_type, model_fun, features, disjoint_cv, n_seed, n_setsel, verbose=True, output_f=None):
    clf= get_classification_model(model_type, model_fun, n_seed)
    all_auc = []
    all_auprc = []
    all_fs = []
    
    for i, (train, test) in enumerate(cv):
        print i
        file_name = None # for saving results
        pairs_train = pairs[train]
        classes_train = classes[train] 
        pairs_test = pairs[test]
        classes_test = classes[test]
	print len(pairs_train), len(pairs_test), len(pairs)
        X = createFeatureMat(pairs_train, classes_train, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames,  featureMatfile=None)
        y = numpy.array(classes_train)
        
        X_new = createFeatureMat(pairs_test, classes_test, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames, featureMatfile=None)
        y_new = numpy.array(classes_test)
	
        probas_ = clf.fit(X, y).predict_proba(X_new)
        y_pred = clf.predict(X_new)
        tn, fp, fn, tp = confusion_matrix(y_new, y_pred).ravel()
        precision = float(tp)/(tp+fp)
        recall = float(tn)/(tn+fp)
        fs=100*float(2*precision*recall/(precision+recall))
        print "True negatives:", tn, "False positives:", fp,"False negatives:", fn, "True positives:",tp
        print "Precision:", precision, "Recall:",recall,"Specifity:",float(tn)/(tn+fp)
        #print "F-Measure",fs

        fpr, tpr, thresholds = roc_curve(y_new, probas_[:, 1]) 
        roc_auc = 100*auc(fpr, tpr)
        all_auc.append(roc_auc)
        prc_auc = 100*average_precision_score(y_new, probas_[:, 1])
        all_auprc.append(prc_auc)
        all_fs.append(fs)
        print "train positive set:",len(y[y==1])," negative set:",len(y[y==0])
        print "test positive set:",len(y_new[y_new==1])," negative set:",len(y_new[y_new==0])
        #print prc_auc
        if verbose:
            print "Fold:", i+1, "# train:", len(pairs_train), "# test:", len(pairs_test), "AUC: %.1f" % roc_auc, "AUPRC: %.1f" % prc_auc, "FScore: %.1f" % fs
 	
    print numpy.mean(all_auc), numpy.std(all_auc), numpy.mean(all_auprc), numpy.std(all_auprc)
    if output_f is not None:
        #output_f.write("n_fold\tn_proportion\tn_setsel\tmodel type\tfeatures\tdisjoint\tauc.mean\tauc.sd\tauprc.mean\tauprc.sd\n")
        output_f.write("%d\t%d\t%d\t%s\t%s\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (n_fold, n_proportion, n_setsel, model_type,  "|".join(features), disjoint_cv, numpy.mean(all_auc), numpy.std(all_auc), numpy.mean(all_auprc), numpy.std(all_auprc), numpy.mean(all_fs), numpy.std(all_fs)))
    return numpy.mean(all_auc), numpy.mean(all_auprc)

def selectFeatureCV( pairs, classes,drugFeatures, diseaseFeatures,  drugFeatureNames, diseaseFeatureNames, cv, n_subset, n_proportion, n_fold, model_type, model_fun, features, disjoint_cv, n_seed, n_setsel, verbose=True, output_f=None):
    clf= get_classification_model(model_type, model_fun, n_seed)
    all_auc = []
    all_auprc = []
    all_fs = []
   
    randomlr = None 
    for i, (train, test) in enumerate(cv):
        print i
        file_name = None # for saving results
        pairs_train = pairs[train]
        classes_train = classes[train] 
        pairs_test = pairs[test]
        classes_test = classes[test]
	print len(pairs_train), len(pairs_test), len(pairs)
        X = createFeatureMat(pairs_train, classes_train, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames,  featureMatfile=None)
        y = numpy.array(classes_train)
        X_new = createFeatureMat(pairs_test, classes_test, drugFeatures, diseaseFeatures, drugFeatureNames, diseaseFeatureNames, featureMatfile=None)
        y_new = numpy.array(classes_test)
       
	if randomlr == None:
	    #feature selection
	    randomlr = load_learn_model('../data/models/randomlr.pkl')
	    selectedFeatures=randomlr.get_support(indices=True)
    	    #print selectedFeatures
    	    saveSelectedFeatures(drugFeatureNames, diseaseFeatureNames, selectedFeatures)
	    #randomlr = linear_model.RandomizedLogisticRegression( C=1, random_state=n_seed, selection_threshold=0.1)
	    #sfm = SelectFromModel(clf)
	    #randomlr.fit(X,y)	
	X = randomlr.transform(X)
	X_new = randomlr.transform(X_new)
	print "number of features seleted",X.shape[1]

        probas_ = clf.fit(X, y).predict_proba(X_new)
        y_pred = clf.predict(X_new)
        tn, fp, fn, tp = confusion_matrix(y_new, y_pred).ravel()
        precision = float(tp)/(tp+fp)
        recall = float(tn)/(tn+fp)
        fs=100*float(2*precision*recall/(precision+recall))
        print "True negatives:", tn, "False positives:", fp,"False negatives:", fn, "True positives:",tp
        print "Precision:", precision, "Recall:",recall,"Specifity:",float(tn)/(tn+fp)
        #print "F-Measure",fs

	fpr, tpr, thresholds = roc_curve(y_new, probas_[:, 1])
        roc_auc = 100*auc(fpr, tpr)
        all_auc.append(roc_auc)
        prc_auc = 100*average_precision_score(y_new, probas_[:, 1])
        all_auprc.append(prc_auc)
        all_fs.append(fs)
        print "train positive set:",len(y[y==1])," negative set:",len(y[y==0])
        print "test positive set:",len(y_new[y_new==1])," negative set:",len(y_new[y_new==0])
        #print prc_auc
        if verbose:
            print "Fold:", i+1, "# train:", len(pairs_train), "# test:", len(pairs_test), "AUC: %.1f" % roc_auc, "AUPRC: %.1f" % prc_auc, "FScore: %.1f" % fs
 	
    print numpy.mean(all_auc), numpy.std(all_auc), numpy.mean(all_auprc), numpy.std(all_auprc)
    if output_f is not None:
        #output_f.write("n_fold\tn_proportion\tn_setsel\tmodel type\tfeatures\tdisjoint\tauc.mean\tauc.sd\tauprc.mean\tauprc.sd\n")
        output_f.write("%d\t%d\t%d\t%s\t%s\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (n_fold, n_proportion, n_setsel, model_type,  "|".join(features), disjoint_cv, numpy.mean(all_auc), numpy.std(all_auc), numpy.mean(all_auprc), numpy.std(all_auprc), numpy.mean(all_fs), numpy.std(all_fs)))
    return numpy.mean(all_auc), numpy.mean(all_auprc)
