import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate,cross_val_score, KFold
from sklearn.metrics import  make_scorer,f1_score
from time import time


def OneHotEncoder__(X):
    start = time()
    cat_ftrs = []
    for col in X.columns:
        if (is_string_dtype(X[col])):
            cat_ftrs.append(col) 
    col_trans = make_column_transformer((OneHotEncoder(sparse=False,handle_unknown='ignore'),cat_ftrs), remainder='passthrough')
    X_trans = col_trans.fit_transform(X)
    done = time()
    elapsed = done - start
    print(type(X_trans))
    print(elapsed)
    return(X_trans)



def adl(X,y,w_solver=None,nb_shrinkage=None,nb_components=None,w_covariance=None,choisis=False,params=None):
    start = time()
    X=OneHotEncoder__(X)
    
    if choisis==False:
        params = {"shrinkage":['auto',0.1,0.3,0.5,0.7,0.9]}
        grid = GridSearchCV(LinearDiscriminantAnalysis(solver='lsqr'),param_grid = params,cv=5)
        fit = grid.fit(X,y)
        best_param = fit.best_params_
        lda = LinearDiscriminantAnalysis(solver='lsqr')
        lda.set_params(**best_param)
    else:
        lda = LinearDiscriminantAnalysis(solver=w_solver,shrinkage=nb_shrinkage,n_components=nb_components,store_covariance=w_covariance)
    
    f1 = cross_val_score(lda,X,y,cv=5,scoring="f1_micro").mean()
    recall = cross_val_score(lda,X,y,cv=5,scoring="recall_micro").mean()
    precision = cross_val_score(lda,X,y,cv=5,scoring="precision_micro").mean()
    y_pred = cross_val_predict(lda, X, y, cv=5)

    done = time()
    elapsed = done - start

    result=dict()
    result["f1_score"]=f1
    result["precision"]= precision
    result["rappel"]= recall
    result["temps"]= elapsed
    result["prediction"]= y_pred
    return(result)


def knnClass(data,X,y,cat=None,params=None,nb_neighbors=None,nweights=None,nmetric=None,nalgorithm=None,choisis = False):
    start = time()
    X=OneHotEncoder__(X)

    if choisis == False:
        params = {'n_neighbors': np.arange(1,10),'weights':['uniform', 'distance'] , 'metric': ['euclidean','manhattan',"minkowski"], 'algorithm':["ball_tree", "kd_tree", "brute"]}
        grid = GridSearchCV(KNeighborsClassifier(),params,cv=5,)
        fit = grid.fit(X,y)
        best_param = fit.best_params_
        knn = KNeighborsClassifier()
        knn.set_params(**best_param)
        
    else:
        knn = KNeighborsClassifier(n_neighbors=nb_neighbors)
    

    f1 = cross_val_score(knn,X,y,cv=5,scoring="f1_micro").mean()
    recall = cross_val_score(knn,X,y,cv=5,scoring="recall_micro").mean()
    precision = cross_val_score(knn,X,y,cv=5,scoring="precision_micro").mean()          
    print(f1, recall, precision)
    y_pred = cross_val_predict(knn, X, y, cv=5)
    
    done = time()
    elapsed = done - start
    
    result=dict()
    result["f1_score"]=f1
    result["precision"]= precision
    result["rappel"]= recall
    result["temps"]= elapsed
    result["prediction"]= y_pred
    return(result)

def knn_reg(X, Y, n_neighbors=5, choisis=False):
    start=time()   
    X=OneHotEncoder__(X)
    if choisis==False :
        modele= KNeighborsRegressor()
        params={'n_neighbors' : [1,2,3,4,5,6,7,8,9,10,11]}
        grille= GridSearchCV(modele,param_grid=params, cv=KFold(n_splits=50,shuffle=True), scoring='r2', n_jobs=-1)
        grille.fit(X,Y)
        knn = KNeighborsRegressor()
        knn.set_params(**grille.best_params_)
     
    else:
        knn=KNeighborsRegressor(n_neighbors=n_neighbors)
        
    knn.fit(X,Y)
  
    Scores = cross_validate(knn, X, Y, scoring=['neg_mean_squared_error','r2'], cv=10)
    y_pred = cross_val_predict(knn, X, Y, cv=5)
   
    done=time()
    elapsed = done - start
    
    result=dict()
    result["temps"]= elapsed
    result["prediction"]=y_pred
    result['R2'] = np.mean(Scores['test_r2'])
    result['mse'] = np.mean(Scores['test_neg_mean_squared_error']) 
    return(result)

def decTree(X, Y, max_depth=None, min_leaf=1, splitter="best", min_samples_split=2, choisis=False): 
    start = time()
    X=OneHotEncoder__(X)

    if choisis==False :
         modele = DecisionTreeRegressor()

         params = {"splitter": ['best','random'], 'min_samples_split':[2,3,4,5,6,7,8]}
         grille = GridSearchCV(modele, param_grid=params, cv=5,scoring='neg_mean_squared_error', n_jobs=-1)
         grille.fit(X,Y)
         dt=DecisionTreeRegressor()
         dt.set_params(**grille.best_params_)
    else:
        dt = DecisionTreeRegressor(splitter=splitter, max_depth=max_depth, min_samples_leaf=min_leaf, min_samples_split=min_samples_split)
    
    dt.fit(X, Y)
  
    Scores = cross_validate(dt, X, Y, scoring=['neg_mean_squared_error','r2'], cv=10)
    
    y_pred = cross_val_predict(dt, X, Y, cv=5)
    
    done = time()
    elapsed = done - start
    
    result=dict()
    result["mso"]=np.mean(Scores['test_neg_mean_squared_error'])
    result["R2"]= np.mean(Scores['test_r2'])
    result["temps"]= elapsed
    result["prediction"]=y_pred
    return(result)

def RegLineaire(X, Y, alpha =1 , l1_ratio=0.5, choisis=False):
    start = time()
    X=OneHotEncoder__(X)
    if choisis==False :
        model = ElasticNet()
        params = {"alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1,1.0, 10.0, 100.0], "l1_ratio": np.arange(0.1, 1, 0.1)}
        grille = GridSearchCV(model, param_grid=params, cv=5,scoring='neg_mean_squared_error', n_jobs=-1)
        grille.fit(X,Y)
        lmodel=ElasticNet()
        lmodel.set_params(**grille.best_params_)
    else:
        lmodel = ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
        
    lmodel.fit(X, Y)
    
    Scores = cross_validate(lmodel, X, Y, scoring=['neg_mean_squared_error','r2'], cv=10)
    y_pred = cross_val_predict(lmodel, X, Y, cv=5)
    
    done = time()
    elapsed = done - start
    
    result=dict()
    result["mso"]=np.mean(Scores['test_neg_mean_squared_error']) 
    result["R2"]= np.mean(Scores['test_r2'])
    result["temps"]= elapsed
    result["prediction"]=y_pred
    return(result)


def regressionLog(X, Y ,solver="saga", l1_ratio= 0, max_iter=5000, choisis=False):
    start = time()
    X=OneHotEncoder__(X)
    
    if len(np.unique(Y))==2:
        if choisis==False:
            scorer=make_scorer(f1_score)
            modele = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=max_iter)
            params = {"l1_ratio": np.arange(0.0, 1.0, 0.1)}
            grille = GridSearchCV(modele, param_grid=params, cv=5,scoring=scorer, n_jobs=-1)
            grille.fit(X,Y)
            
            regression_logistique = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=max_iter)
            regression_logistique.set_params(**grille.best_params_)
        else :
            regression_logistique = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=l1_ratio, max_iter=max_iter)
            #Instanciation du modèle
   
        regression_logistique.fit(X, Y)
        
        Scores = cross_validate(regression_logistique, X, Y, scoring =['f1','precision','recall'] , cv=3)
        score =np.mean(Scores['test_f1'])
        precision = np.mean(Scores['test_precision'])
        recall= np.mean(Scores['test_recall'])
        
    else:   
        #MULTICLASS
        scorer= make_scorer(f1_score, average='macro')
        if choisis==False:

            modele = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=max_iter,multi_class='multinomial')
            params = {"l1_ratio": np.arange(0.0, 1.0, 0.1)}
            grille = GridSearchCV(modele, param_grid=params, cv=5,scoring=scorer, n_jobs=-1)
            grille.fit(X,Y)

            #On a trouvé les meilleurs paramètres, on intialise une regression logistique avec ces paramètres
            regression_logistique = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=max_iter, multi_class='multinomial')
            regression_logistique.set_params(**grille.best_params_)
            
        else:
            regression_logistique = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=l1_ratio, max_iter=max_iter) #Instanciation du modèle
        
        #On fit notre regression sur les données
        regression_logistique.fit(X,Y)
        
        Scores = cross_validate(regression_logistique, X, Y, scoring =['f1_macro', 'precision_macro', 'recall_macro'] , cv=5)
        score= np.mean(Scores['test_f1_macro'])
        precision =np.mean(Scores['test_precision_macro'])
        recall=np.mean(Scores['test_recall_macro'])
        y_pred = cross_val_predict(regression_logistique, X, Y, cv=5)
    done = time()
    elapsed = done - start
    
    result=dict()
    result["f1_score"]=score
    result["precision"]= precision
    result["rappel"]= recall
    result["temps"]= elapsed
    result["y_pred"]=y_pred
    return(result)


def svr_reg(X,Y, kernel='rbf', gamma=0.001, C=0.5, degree=2, choisis=False):
    start=time()
    X=OneHotEncoder__(X)
    #standardiser
    scal = StandardScaler(copy=False)
    scal.fit(X)
    X= scal.transform(X)
    Y=np.ravel(Y)
    
    if choisis==False :
        modele= SVR()
        params={'kernel':['linear', 'rbf','sigmoid','poly'],
                 'gamma': [1e-7, 1e-10, 1e-4,0.001, 0.01, 0.1,0.5],
                 'C' : [0.1,0.5,1,3,5,10,30,70],
                 'degree' : [0.01,0.2,0.5,1,2,3,4]}
        grille= GridSearchCV(modele,param_grid=params, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
        grille.fit(X,Y)
        svr = SVR()
        svr.set_params(**grille.best_params_)
    else:
        svr=SVR(kernel=kernel, gamma=gamma, degree=degree, C=C)
        
    svr.fit(X,Y)
    
    Scores = cross_validate(svr, X, Y, scoring=['neg_mean_squared_error','r2'], cv=10)

    y_pred = cross_val_predict(svr, X, Y, cv=5)
    done=time()
    elapsed = done - start
    
    result=dict()
    result["temps"]= elapsed
    result["prediction"]=y_pred
    result['R2'] = np.mean(Scores['test_r2'])
    result['mse'] = np.mean(Scores['test_neg_mean_squared_error']) 
    return(result)
