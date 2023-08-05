#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""The module performs 5-fold nested CV on the Baseline dataset using ML and clinical models."""
# =============================================================================
# Imports
from data_preparation import prepare_patient_dataset
from constants import data_dir, instruction_dir, discrepency_dir
from data_preparation import get_abb_to_long_dic

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from clinical_score import ClinicalScore
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, average_precision_score, roc_auc_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector


import pandas as pd
import pickle 
import time
import sys
import itertools
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import completeness_score, homogeneity_score
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster import hierarchy

all_models = {
    'CHAP':"Clinical",
    'ACCP':'Clinical',
    'RIETE':'Clinical',
    'VTE-BLEED':'Clinical',
    'HAS-BLED':'Clinical',
    'OBRI':'Clinical',
    
    'LogisticRegression': "ML",
    "LDA": "ML",
    "QDA": "ML",
    "SVC": "ML",
    "GaussianNB": "ML",
    "RandomForest": "ML",
    "AdaBoost": "ML",
    "GradientBoosting" : "ML",
    "Dummy": "ML"
}

nice_names = {
    'CHAP':"CHAP",
    'ACCP':"ACCP",
    'RIETE':"RIETE",
    'VTE-BLEED':"VTE-BLEED",
    'HAS-BLED':"HAS-BLED",
    'OBRI':"OBRI",
    
    'LogisticRegression': "Logistic\nRegression",
    "LDA": "LDA",
    "QDA": "QDA",
    "SVC": "SVC",
    "GaussianNB": "Gaussian\nNB",
    "RandomForest": "Random\nForest",
    "AdaBoost": "Ada\nBoost",
    "GradientBoosting" : "Gradient\nBoosting",
    "Dummy":"Dummy",
}

def my_custom_metrics(y_true, y_pred, metric):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if metric == "tp":
        return tp
    if metric == "tn":
        return tn
    if metric == "fp":
        return fp
    if metric == "fn":
        return fn
    if metric == "specificity":
        return tn/(tn+fp)

def count_length_of_grid_search():
    """Prints the number of grid search space for the ML models.
    """
    
    def get_hyperparameter_combinations_count(param_grid_list):
    
        lists_of_iter = []
        for param_grid in param_grid_list:
            for i in itertools.product(*param_grid.values()):
                lists_of_iter.append(i)
        return len(lists_of_iter)
    
    for model_name in all_models:
        if all_models[model_name] == "ML":
            model, param_grid = get_param_grid_model(model_name)

            print(model_name, get_hyperparameter_combinations_count(param_grid))   
        

def generate_metric_pictures():
    
    #Creates the graphs showing metrics average in 10-fold-nested cv
    def create_graph(ml_data_df, metric_name):
        fig, ax = plt.subplots(figsize=(10.5,5))
        
        ml_data = ml_data_df.copy()
        
        ml_data.loc[:,"model"] = ml_data["model"].apply(lambda x:nice_names[x])

        #Set the size of the fonts
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.title_fontsize'] = 13

        #Order the MLs based on the mean of their score
        order = {ml_model:ml_data[ml_data['model']==ml_model]["value"].mean() for ml_model in ml_data['model'].unique()}
        ordered_xlabels = [k for k, v in sorted(order.items(),reverse=True, key=lambda item: item[1])]

        sns.boxplot(data=ml_data, x="model", y="value", ax=ax, linewidth=1, color="white", 
                    flierprops={"marker": None},
                    boxprops={"edgecolor": "black"},
                    width=0.4, order=ordered_xlabels)
        
        # Change the colirs of whiskers
        for i, patch in enumerate(ax.patches):
            for line in ax.lines[i * 6: (i + 1) * 6]:
                line.set_color("black")

        sns.swarmplot(data=ml_data, x="model", y="value", ax=ax, size=5, color="gray", alpha=0.6, order=ordered_xlabels)

        for x_axis, ml_model in enumerate(ordered_xlabels):
            mean = ml_data[ml_data['model']==ml_model]["value"].mean()
            max_value =  ax.get_ylim()[1]
            ax.text(x_axis-0.35, max_value+(0.02*max_value), s=f"μ={mean:.2f}", size=8)

        # ax.spines["right"].set_visible(False)
        # ax.spines["top"].set_visible(False)
        
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_title(metric_name, pad=30, fontsize=18)
        ax.set_xlabel("Models", fontsize=14)
        ax.set_ylabel("Values", fontsize=14)
        
        xlabel_to_type_dic = {v:all_models[k] for k,v in nice_names.items()}
        
        for i, model in enumerate(ordered_xlabels):
            if xlabel_to_type_dic[model]=="ML":
                plt.setp(ax.get_xticklabels()[i], color='red')

        
        fig.savefig(f"./sklearn_results/Figures/{metric_name}.pdf", bbox_inches="tight")
        fig.savefig(f"./sklearn_results/Figures/{metric_name}.png", dpi=500, bbox_inches="tight")

    #Populate the metrics data into a dic
    all_model_metrics = dict()

    #The metrics of interests
    metrics = ["AUPRC", "AUROC"]

    #Populate the dic
    for model in all_models:
        if all_models[model] == "ML":

            model_metrics = dict()

            with open(f"./sklearn_results/{model}_nested_cv_results.pickle", 'rb') as handle:
                data = pickle.load(handle)

            for metric in metrics:
                model_metrics[metric] = list(data[f"test_{metric}"])
                
            all_model_metrics[model] = model_metrics
                
        elif all_models[model] == "Clinical":
            
            data = pd.read_csv(f"./sklearn_results/{model}_nested_cv_results.csv").drop("Unnamed: 0", axis=1).to_dict(orient='list')

            all_model_metrics[model] = data
        
    #Reformat the data into df   
    df_all = pd.DataFrame()
    for model in all_model_metrics.keys():

        df = pd.DataFrame.from_dict(all_model_metrics[model])
        df["model"] = model
        df_all = pd.concat([df_all, df])

    number_cvs = int(len(df_all)/len(all_models))
    
    df_all["test_set"] = [f'test-split-{i+1}' for i in range(number_cvs)]*len(all_models)

    df_all = df_all.melt(id_vars=['model', "test_set"])

    #For each metric, draw the graphs 
    for metric in metrics:
    
        data = df_all[df_all["variable"]==metric]
        create_graph(data, metric)
    
    #############################################################
    #Draw the best feature selection method
    feature_selection_counter = dict()

    for model in [model for model in all_models if all_models[model]=="ML" if model != "Dummy"]:
        with open(f"./sklearn_results/{model}_nested_cv_results.pickle", 'rb') as handle:
            nested_score = pickle.load(handle)
            
        model = nice_names[model]
        
        feature_selection_counter[model] = {"PCA":0, "SelectFromModel":0, "None":0}
        
        for i in range(len(nested_score['estimator'])):
            est1 = nested_score['estimator'][i]
            feature_selection_method = est1.best_estimator_.steps[1][1]
            reduce_dim_method = est1.best_estimator_.steps[2][1]
        
            if feature_selection_method == reduce_dim_method: #Means both are passthrough, and no reduction in # of features occured
                feature_selection_counter[model]["None"] += 1
            elif reduce_dim_method != "passthrough": #Means PCA has occured
                feature_selection_counter[model]["PCA"] += 1
            else:#Means Select from model occured
                feature_selection_counter[model]["SelectFromModel"] += 1
            
    fig, ax = plt.subplots(figsize=(8, 4))
    

    feat_select_df = pd.DataFrame.from_dict(feature_selection_counter).T
    feat_select_df["model"] = feat_select_df.index
    feat_select_df = feat_select_df.melt(id_vars ="model", value_vars=["PCA", "SelectFromModel", "None"], var_name="method")

    sns.barplot(data=feat_select_df, x="model", y="value", hue="method", ax=ax, palette="icefire", width=0.6)
    ax.legend(bbox_to_anchor=(0.5, 0.85), loc='center left')
    ax.set_xlabel("Models", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)

    fig.savefig(f"./sklearn_results/Figures/best_selection_methods.pdf", bbox_inches="tight")
    fig.savefig(f"./sklearn_results/Figures/best_selection_methods.png", dpi=300, bbox_inches="tight")
            


def get_param_grid_model(classifier):
    
    #Params for sequential feature selection
    sfs_scoring = "average_precision"
    sfs_cv = 3

    #Params for grid search
    C_values = [0.01, 0.1, 1, 10, 50]
    class_weights = [None, "balanced"]
    
    
    if classifier=="LogisticRegression":

        param_grid= [
            {
            'reduce_dim': [PCA(5), PCA(10)],
            'model__class_weight': class_weights,
            'model__C': C_values,
            'model__penalty': ["l1", "l2",],
            },
            
            {
            'model__C': C_values,
            'model__penalty': ["l1", "l2",],
            'model__class_weight': class_weights,
            },   
            
            {
            'feature_selection': [SequentialFeatureSelector(LogisticRegression(class_weight="balanced", random_state=1), 
                                                                n_features_to_select=5, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  
                                  SequentialFeatureSelector(LogisticRegression(class_weight="balanced", random_state=1), 
                                                                n_features_to_select=10, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  ],
            'model__C': C_values,
            'model__penalty': ["l1", "l2",],
            'model__class_weight': class_weights,
            },            
            ]
        
        model = LogisticRegression(random_state=1, solver="liblinear", max_iter=10000)
        
    
    elif classifier == "LDA":
        param_grid= [
            {"model__shrinkage":[None, 0.1, 0.5, 0.9, 'auto']},  
            
            {"reduce_dim": [PCA(5), PCA(10)],
            "model__shrinkage":[None, 0.1, 0.5, 0.9, 'auto']},
            
            {'feature_selection':[SequentialFeatureSelector(LinearDiscriminantAnalysis(), 
                                                                n_features_to_select=5, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  
                                  SequentialFeatureSelector(LinearDiscriminantAnalysis(), 
                                                                n_features_to_select=10, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  ],
             "model__shrinkage":[None, 0.1, 0.5, 0.9, 'auto']
             }
                 
                     ]
        
        model = LinearDiscriminantAnalysis()
        
        
    
    elif classifier == "QDA":
        param_grid= [
            {"model__reg_param":[0, 0.1, 0.5, 0.9]},    
            
            {"reduce_dim": [PCA(5), PCA(10)],
             "model__reg_param":[0, 0.1, 0.5, 0.9]},
            
            {'feature_selection':[SequentialFeatureSelector(QuadraticDiscriminantAnalysis(), 
                                                                n_features_to_select=5, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  
                                  SequentialFeatureSelector(QuadraticDiscriminantAnalysis(), 
                                                                n_features_to_select=10, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  ],
             "model__reg_param":[0, 0.1, 0.5, 0.9]}            
                     ]
        
        model = QuadraticDiscriminantAnalysis()
    
    elif classifier == "GaussianNB":
        param_grid= [
            {
                'feature_selection': [SequentialFeatureSelector(GaussianNB(), 
                                                                n_features_to_select=5, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  
                                  SequentialFeatureSelector(GaussianNB(), 
                                                                n_features_to_select=10, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  ],
                'model__var_smoothing': [1e-11 ,1e-10, 1e-9, 1e-8, 1e-7],
            },
            
            {'reduce_dim': [PCA(5), PCA(10)],
             'model__var_smoothing': [1e-11 ,1e-10, 1e-9, 1e-8, 1e-7]},
            
            {'model__var_smoothing': [1e-11 ,1e-10, 1e-9, 1e-8, 1e-7]}
        ]   
        
        model = GaussianNB()
        
    
    elif classifier == "SVC":
        param_grid= [{
            'reduce_dim': [PCA(5), PCA(10)],
            'model__kernel': ["rbf", "linear"],
            'model__C': C_values,
            'model__gamma': ['auto', 'scale', 0.01, 0.1, 1, 10],
            'model__class_weight': class_weights
            },
                     
            {
            'feature_selection': [SequentialFeatureSelector(SVC(random_state=1, class_weight='balanced'), 
                                                                n_features_to_select=5, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  
                                  SequentialFeatureSelector(SVC(random_state=1, class_weight='balanced'), 
                                                                n_features_to_select=10, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  ],
            'model__kernel': ["rbf", "linear"],
            'model__C': C_values,
            'model__gamma': ['auto', 'scale', 0.01, 0.1, 1, 10],
            'model__class_weight': class_weights
            },
            
            {
            'model__kernel': ["rbf", "linear"],
            'model__C': C_values,
            'model__gamma': ['auto', 'scale', 0.01, 0.1, 1, 10],
            'model__class_weight': class_weights
            }

            ]
        
        model = SVC(random_state=1)
        
        
    elif classifier == "RandomForest":
        rf_max_features = ['sqrt', 'log2', None, 0.3]
        rf_max_depth = [10, 20, 30, 90, None]
        rf_n_estimators = [10, 50, 100, 150]
        rf_min_sample_split = [2, 5, 10]
        
        param_grid= [{
                     'reduce_dim' : [PCA(5), PCA(10)],
                     'model__max_depth': rf_max_depth,
                     'model__max_features': rf_max_features,
                     'model__min_samples_split': rf_min_sample_split,
                     'model__n_estimators': rf_n_estimators,
                     'model__class_weight': class_weights},
                     
                     {
                    'feature_selection': [SequentialFeatureSelector(RandomForestClassifier(random_state=1, n_jobs=-1, class_weight="balanced"), 
                                                                n_features_to_select=5, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  
                                          SequentialFeatureSelector(RandomForestClassifier(random_state=1, n_jobs=-1, class_weight="balanced"), 
                                                                n_features_to_select=10, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  ],
                     'model__max_depth': rf_max_depth,
                     'model__max_features': rf_max_features,
                     'model__min_samples_split': rf_min_sample_split,
                     'model__n_estimators': rf_n_estimators,
                     'model__class_weight': class_weights},
                     
                    {
                     'model__max_depth': rf_max_depth,
                     'model__max_features': rf_max_features,
                     'model__min_samples_split': rf_min_sample_split,
                     'model__n_estimators': rf_n_estimators,
                     'model__class_weight': class_weights}
                     ]
        
        model = RandomForestClassifier(random_state=1, n_jobs=-1)
        
    elif classifier == "AdaBoost":
        param_grid= [{
            'reduce_dim': [PCA(5), PCA(10)],
            'model__learning_rate': [0.1, 0.5, 1, 10],
            'model__n_estimators': [1, 5, 10, 20, 50, 100],
            
        },
        {
            'feature_selection': [SequentialFeatureSelector(AdaBoostClassifier(random_state=1), 
                                                                n_features_to_select=5, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  
                                          SequentialFeatureSelector(AdaBoostClassifier(random_state=1), 
                                                                n_features_to_select=10, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  ],
            'model__learning_rate': [0.1, 0.5, 1, 10],
            'model__n_estimators': [1, 5, 10, 20, 50, 100],
            
        },
        {
            'model__learning_rate': [0.1, 0.5, 1, 10],
            'model__n_estimators': [1, 5, 10, 20, 50, 100],
            
        },       
        
        ]
        
        model = AdaBoostClassifier(random_state=1)
    
    elif classifier == "GradientBoosting":
        
        param_grid = [
            {
                'reduce_dim': [PCA(5), PCA(10)],
                'model__learning_rate':[0.1, 0.2, 0.3, 0.5],
                'model__n_estimators': [10, 50, 150],
                'model__subsample': [0.1, 0.2, 0.4, 1.0],
                'model__min_samples_split': [2, 5],
                'model__max_depth': [3, 5, None],
                'model__max_features': ['sqrt', 'log2', None]
            },
            
            {
                'reduce_dim': [SequentialFeatureSelector(GradientBoostingClassifier(random_state=1), 
                                                                n_features_to_select=5, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  
                                          SequentialFeatureSelector(GradientBoostingClassifier(random_state=1), 
                                                                n_features_to_select=10, scoring=sfs_scoring, 
                                                                cv=sfs_cv, direction='forward'),
                                  ],
                'model__learning_rate':[0.1, 0.2, 0.3, 0.5],
                'model__n_estimators': [10, 50, 150],
                'model__subsample': [0.1, 0.2, 0.4, 1.0],
                'model__min_samples_split': [2, 5],
                'model__max_depth': [3, 5, None],
                'model__max_features': ['sqrt', 'log2', None]
            },
            
            {
                'model__learning_rate':[0.1, 0.2, 0.3, 0.5],
                'model__n_estimators': [10, 50, 150],
                'model__subsample': [0.1, 0.2, 0.4, 1.0],
                'model__min_samples_split': [2, 5],
                'model__max_depth': [3, 5, None],
                'model__max_features': ['sqrt', 'log2', None]
            }
        ]
        
        model = GradientBoostingClassifier(random_state=1)
    
        
        
    elif classifier == "Dummy":
        param_grid = [{"model__strategy": ["stratified", "most_frequent"]}]
        model = DummyClassifier()
        
    
    #In order to reduce redundancy in code, we add the proprocessing beforehand.
    for item in param_grid:
        item['preprocess'] = [StandardScaler()]

                                      
        
    pipe = Pipeline([('preprocess', 'passthrough'),
                     ('feature_selection', 'passthrough'),
                    ('reduce_dim','passthrough'),
                    ('model', model)],
                    memory="$SLURM_TMPDIR")
    
    return pipe, param_grid


def perform_nested_cv(model, X, y):
    
    #Custom Scores
    tn_score = make_scorer(my_custom_metrics, greater_is_better=True, metric="tn")
    tp_score = make_scorer(my_custom_metrics, greater_is_better=True, metric="tp")
    fn_score = make_scorer(my_custom_metrics, greater_is_better=True, metric="fn")
    fp_score = make_scorer(my_custom_metrics, greater_is_better=True, metric="fp")
    specificity_score = make_scorer(my_custom_metrics, greater_is_better=True, metric="specificity")

    
    #Copy the input X and the output y  
    baseline_dataframe_duplicate = X.copy()
    target_series_duplicate = y.copy()
    
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    
    print(f'Started performing nested cv for {model}')
    start_time = time.time()
    
    if all_models[model] == "ML":
        
        pipe, param_grid = get_param_grid_model(model)



        clf = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
        # clf = RandomizedSearchCV(estimator=pipe, n_iter=10, param_distributions=param_grid, cv=inner_cv, scoring="roc_auc")
        nested_score = cross_validate(clf, X=baseline_dataframe_duplicate, n_jobs=-1,
                                      return_estimator=True, y=target_series_duplicate, cv=outer_cv, 
                                    scoring={"Accuracy":'accuracy',
                                            "Sensitivity":'recall',
                                            "AUPRC":'average_precision',
                                            "AUROC":'roc_auc',
                                            "Precision":make_scorer(precision_score,zero_division=0),
                                            "F1-score":'f1',
                                            "Specificity": specificity_score,
                                            "TN":tn_score,
                                            "TP":tp_score,
                                            "FN":fn_score,
                                            "FP":fp_score,
                                            })
        
        with open(f"./sklearn_results/{model}_nested_cv_results.pickle", 'wb') as handle:
            pickle.dump(nested_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    elif all_models[model] == "Clinical":
        #Create the score object
        clinical_model = ClinicalScore(model)

        #Fitting doesn't do anything special, it is just required for the class
        clinical_model = clinical_model.fit(baseline_dataframe_duplicate, target_series_duplicate)

        #Add the uniqid as a column to the baseline data because the split will re-index the data
        baseline_dataframe_duplicate["uniqid"] = baseline_dataframe_duplicate.index
        
        #cv_dictionary
        cv_results_dic = dict()
        
        #For each test index set, compute the metrics
        for i, (train_index, test_index) in enumerate(outer_cv.split(baseline_dataframe_duplicate,
                                                                     target_series_duplicate)):
            
            X_test_uniqids = list(baseline_dataframe_duplicate.iloc[test_index]['uniqid'])
            X_test_data = baseline_dataframe_duplicate.loc[X_test_uniqids]
            y_test = target_series_duplicate.loc[X_test_uniqids]

            #Record exactly what are the predictions for each sample on the test dataset
            y_pred_classes = clinical_model.predict(X_test_data) #This is the class that the model predicts
            y_pred_proba = clinical_model.predict(X_test_data, mode="score") #This is the score (similar to probability) of the model
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_classes).ravel()
            specificity = tn/(tn+fp)

            record_dict = {"Accuracy":accuracy_score(y_test, y_pred_classes),
                           "Precision":precision_score(y_test, y_pred_classes, zero_division=0),
                           "Sensitivity":recall_score(y_test, y_pred_classes),
                           "AUROC":roc_auc_score(y_test, y_pred_proba), #The probability is used instead of the threshold-decided class
                           "AUPRC":average_precision_score(y_test, y_pred_proba), #The probability is used instead of the threshold-decided class
                           "F1-score": f1_score(y_test, y_pred_classes),
                           "Specificity": specificity,
                           "TN":tn,
                            "TP":tp,
                            "FN":fn,
                            "FP":fp
                           }
            
            cv_results_dic[f"test_split_{i}"] = record_dict

        #Save the detailed results
        pd.DataFrame(cv_results_dic).T.to_csv(f"./sklearn_results/{model}_nested_cv_results.csv")


    print(model, "is done in", f'{time.time()-start_time}', "seconds.")


def main(mode):
    #Read the patient dataset
    patient_dataset = prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir)

    #Remove patients without baseline, remove the FUPS after bleeding/termination, fill FUP data for those without FUP data
    patient_dataset.filter_patients_sequentially(mode="fill_patients_without_FUP")
    print(patient_dataset.get_all_targets_counts(), "\n")

    #Add one feature to each patient indicating year since baseline to each FUP
    patient_dataset.add_FUP_since_baseline()

    #Get the BASELINE, and Follow-up data from patient dataset
    _, _, baseline_dataframe, target_series = patient_dataset.get_data_x_y(baseline_filter=["uniqid", "dtbas", "vteindxdt", "stdyoatdt"], 
                                                                                                FUP_filter=[])
    
    #Get the genotype data
    genotype_data = patient_dataset.get_genotype_data()

    #Change the names of the columns in the baseline data to make them compatible with the clinical score
    abb_to_long_dic = get_abb_to_long_dic(instructions_dir=instruction_dir, CRF_name="BASELINE")
    baseline_dataframe.columns = [abb_to_long_dic[col] if col in abb_to_long_dic else col for col in baseline_dataframe.columns]
    

    #Concat the Baseline and Genotype data
    concat_x = baseline_dataframe.join(genotype_data)


    #Perform nested cv
    if mode == "all_models":
        for model in all_models:
            perform_nested_cv(model, concat_x, target_series)
    elif mode in all_models.keys():
        perform_nested_cv(mode, concat_x, target_series)
    elif mode == "test":
        print("Data preparation was finished succesfully.")
    else:
        raise Exception(f"The mode {mode} doesn't exist.")
    

def perform_unsupervised_learning():
    
    #Function to get the full (descriptive) name of the features
    def get_long_name(abb_to_long_dictionary, col):
        if "_" in col:
            return abb_to_long_dictionary[col.split("_")[0]]+"-"+col.split("_")[1]
        elif col in abb_to_long_dictionary:
            return abb_to_long_dictionary[col]
        else:
            return col
        
    #Read the patient dataset
    patient_dataset = prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir)

    #Remove patients without baseline, remove the FUPS after bleeding/termination, fill FUP data for those without FUP data
    patient_dataset.filter_patients_sequentially(mode="fill_patients_without_FUP")
    print(patient_dataset.get_all_targets_counts(), "\n")

    #Add one feature to each patient indicating year since baseline to each FUP
    patient_dataset.add_FUP_since_baseline()

    #Get the BASELINE, and Follow-up data from patient dataset
    _, _, baseline_dataframe, target_series = patient_dataset.get_data_x_y(baseline_filter=["uniqid", "dtbas", "vteindxdt", "stdyoatdt"], 
                                                                                                FUP_filter=[])

    #Get the genotype data
    genotype_data = patient_dataset.get_genotype_data()

    #Change the names of the columns in the baseline data to make them compatible with the clinical score
    abb_to_long_dic = get_abb_to_long_dic(instructions_dir=instruction_dir, CRF_name="BASELINE")
    baseline_dataframe.columns = [get_long_name(abb_to_long_dic, col) for col in baseline_dataframe.columns]


    #Concat the Baseline and Genotype data
    concat_x = baseline_dataframe.join(genotype_data)
    
    
    ###########################
    #Calculate the time since bleeding from baseline and categorize patients
    
    bleeding_since_baseline = []
    for patient in patient_dataset.all_patients:
        if patient.AD1 is not None:
            number_of_maj_bleed = 0
            for bleed_occurance in patient.AD1:
                if bleed_occurance["majbldconf"] == 1:
                    number_of_maj_bleed += 1
                    date_of_bleed = bleed_occurance["blddtadj"]
                    date_of_baseline = patient.BASELINE["dtbas"].values[0]
                    year_since_bleed = round(pd.Timedelta(date_of_bleed - date_of_baseline).days/365., 2)
                    if year_since_bleed <= 1:
                        year_since_bleed_str = "Bleeders (time-since-baseline ≤ 1-year)"
                    elif (year_since_bleed > 1) and (year_since_bleed <= 2):
                        year_since_bleed_str = "Bleeders (1-year < time-since-baseline ≤ 2-years)"
                    else:
                        year_since_bleed_str = "Bleeders (time-since-baseline > 2-years)"
                    
                    #Some patients have more than one major bleed, we only keep the first one (hence the if statement)
                    if number_of_maj_bleed == 1:
                        bleeding_since_baseline.append(year_since_bleed_str)
            #There are patients who have the AD1 filled for them, but they didn't bleed
            if number_of_maj_bleed == 0:
                bleeding_since_baseline.append("Non-bleeders")

        else:
            bleeding_since_baseline.append("Non-bleeders")
            
    ###################################
    
    
    #Scale the baseline dataset for unsupervised learning
    scaler = StandardScaler()
    concat_scaled_df = pd.DataFrame(scaler.fit_transform(concat_x),  columns = concat_x.columns)
    
    
    #Dimentionality reduction techniques
    feature_decomp_method_dics = {
        "PCA": PCA(n_components=2),
        "Kernel PCA": KernelPCA(n_components=2, kernel="rbf",  gamma=None, fit_inverse_transform=False, alpha=1),
        "Isomap": Isomap(n_components=2, n_neighbors=20, p=1),

        "t-SNE\n(perplexity: 5)": TSNE(n_components=2, perplexity=5, init="random", n_iter=250, random_state=0),
        "t-SNE\n(perplexity: 10)": TSNE(n_components=2, perplexity=10, init="random", n_iter=250, random_state=0),
        "t-SNE\n(perplexity: 15)": TSNE(n_components=2, perplexity=15, init="random", n_iter=250, random_state=0),
        "t-SNE\n(perplexity: 30)": TSNE(n_components=2, perplexity=30, init="random", n_iter=250, random_state=0),
    }

    # y = ["bleeders" if val==1 else "non-bleeders" for val in target_series ]
    # colors = ["red" if val==1 else "blue" for val in target_series ]
    
    #######################################
    #Figure with lower dimention visualized

    colors = ["#ed6925" if status != "Non-bleeders" else "#000004" for status in bleeding_since_baseline]
    alphas = [1 if status != "Non-bleeders" else 0.5 for status in bleeding_since_baseline]
    zorders = [3 if status != "Non-bleeders" else -1 for status in bleeding_since_baseline ]
    markers = ["o" if status != "Non-bleeders" else "o" for status in bleeding_since_baseline ]
    edgecolors = [None if status != "Non-bleeders" else None for status in bleeding_since_baseline ]
    labels = ["Bleeders" if status != "Non-bleeders" else "Non-bleeders" for status in bleeding_since_baseline ]

    def legend_without_duplicate_labels(ax):
        ax.legend( fontsize = 6)

        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='lower left', bbox_to_anchor=(0, 1.2, 1, 0.2),
            ncol=3, mode="expand", fontsize=12)


    fig = plt.figure(layout="constrained", figsize=(8.3, 5.6))
    subfigs = fig.subfigures(2, 1, hspace= 0.05, wspace= 0.05)

    axsTop = subfigs[0].subplots(1, 3)
    axsBottom = subfigs[1].subplots(1, 4)
        

    for method, ax in zip(feature_decomp_method_dics, [*axsTop, *axsBottom]):
        
        X_processed = feature_decomp_method_dics[method].fit_transform(concat_scaled_df)
        

        for i, (color, alpha, zorder, label, marker, edgecolor) in enumerate(zip(colors, alphas, zorders, labels, markers, edgecolors)):
            ax.scatter(X_processed[i,0], X_processed[i,1], marker=marker, c = color, s=12, linewidth=0.1, label=label, zorder= zorder, alpha=alpha)

        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        ax.set_title(method, fontsize=15)

    legend_without_duplicate_labels(axsTop[1])

    fig.savefig(f"./sklearn_results/Figures/dimentionality_reduction_all.pdf",  bbox_inches='tight')  

    ####################################
    
    ###################################
    #Perform clusterting using K-means and Agglomerative clustering on all of the dataset
    
    legend_elements = [Line2D([0], [0], marker='o', color='black', markeredgecolor="black" ,lw=0, markersize=12, label='Non-bleeders'),
                   Line2D([0], [0], marker='X', color='black', markeredgecolor="yellow",lw=0, label='Bleeders', markersize=12),
                   Patch(facecolor="#f98e09",  label='Cluster 1'),
                   Patch(facecolor="#57106e", label='Cluster 2'),
    ]

    clustering_methods_dic = {
        "K-Means": KMeans(n_clusters=2, random_state=0, n_init="auto"),
        "Agglomerative Clustering": AgglomerativeClustering(linkage="ward", n_clusters=2)
    }


    X_PCA = PCA(n_components=2).fit_transform(concat_scaled_df)


    alphas = [1 if status != "Non-bleeders" else 1 for status in bleeding_since_baseline]
    zorders = [3 if status != "Non-bleeders" else -1 for status in bleeding_since_baseline ]
    markers = ["X" if status != "Non-bleeders" else "o" for status in bleeding_since_baseline ]
    edgecolors = ["yellow" if status != "Non-bleeders" else None for status in bleeding_since_baseline ]


    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'hspace': 0.15, 'wspace': 0.05})

    for method, ax in zip(clustering_methods_dic, axs.flat):



        clusters = clustering_methods_dic[method].fit(concat_scaled_df)
        
        colors = ["#f98e09" if label != 0 else "#57106e" for label in clusters.labels_]


        for i, (color, alpha, zorder, label, marker, edgecolor) in enumerate(zip(colors, alphas, zorders, bleeding_since_baseline, markers, edgecolors)):
            ax.scatter(X_PCA[i,0], X_PCA[i,1], marker=marker, c = color, s=14, linewidth=0.1, label=label, zorder= zorder, 
                    alpha=alpha, edgecolors=edgecolor)

        # legend_without_duplicate_labels(ax)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        ax.text(0.01, 0.99, ha='left', va='top',transform=ax.transAxes,size=12,
                s=f"Homogeneity: {homogeneity_score(target_series ,clusters.labels_):.2e} \nCompleteness: {completeness_score(target_series,clusters.labels_):.2e}")

        ax.set_title(method, fontsize=16)
        
    axs[1].legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.04, 0.5), fontsize=12)
        
    fig.savefig(f"./sklearn_results/Figures/clustering_pic_all_points.pdf",  bbox_inches='tight')  
    ###########################################
    
    #######################################
    #Clustering and PCA just on the bleeders
    #Just analyse the bleeders now
    only_bleeders_X = concat_x[target_series==1]
    only_bleeders_X_scaled = pd.DataFrame(StandardScaler().fit_transform(only_bleeders_X), columns=only_bleeders_X.columns, index=only_bleeders_X.index)    
    
    #########
    #The agglomerative clustering on the bleeders
    fig, ax = plt.subplots(figsize=(15,5))

    Z = linkage(only_bleeders_X_scaled, 'ward')

    hierarchy.set_link_color_palette(["#440154",'#5ec962', '#f98e09'])

    a = dendrogram(Z, labels=only_bleeders_X_scaled.index, leaf_rotation=90, ax=ax)

    max_d = 29


    ax.axhline(y=max_d, xmin = 0, xmax=1000, color='gray', linestyle="dotted")

    ax.get_xaxis().set_visible(False)

    ax.set_ylabel("Distance", fontsize=20)
    ax.yaxis.set_tick_params(labelsize=15)

    ax.text(x=10, y=30, s=f"Cutoff: {max_d}", size=13)

    fig.savefig(f"./sklearn_results/Figures/agglomerative_clustering_bleeders.pdf",  bbox_inches='tight')  
    ##########
    
    ###################
    #Create PCA of bleeders and visualize it with the three groups found before
    #Draw loading vectors
    
    #Clusters as defined by the agglomerative clustering
    clusters = fcluster(Z, max_d, criterion='distance')

    #PCA object
    pca_obj = PCA(n_components=2).fit(only_bleeders_X_scaled)

    #PCA results
    X_pca = pca_obj.fit_transform(only_bleeders_X_scaled)

    #Get the loadings of the pca object
    loadings = pd.DataFrame(pca_obj.components_.T, columns=['PC1', 'PC2'], index=only_bleeders_X_scaled.columns)
    # loadings["sum"] = loadings.apply(lambda x: (np.abs(x["PC1"])+np.abs(x["PC2"])), axis=1)
    # loadings = loadings.sort_values(by="sum", ascending=False)

    #Scaling factor to draw the loading vectors
    scale_PC1 = 1.0/(X_pca[:,0].max() - X_pca[:,0].min())
    scale_PC2 = 1.0/(X_pca[:,1].max() - X_pca[:,1].min())
    
    #Clusters as defined by the agglomerative clustering
    clusters = fcluster(Z, max_d, criterion='distance')

    #PCA object
    pca_obj = PCA(n_components=2).fit(only_bleeders_X_scaled)

    #PCA results
    X_pca = pca_obj.fit_transform(only_bleeders_X_scaled)

    #Get the loadings of the pca object
    loadings = pd.DataFrame(pca_obj.components_.T, columns=['PC1', 'PC2'], index=only_bleeders_X_scaled.columns)
    #Save all the loadings just in case
    loadings.to_excel("./sklearn_results/Figures/bleeders_PCA-loadings_all.xlsx")

    #Scaling factor to draw the loading vectors
    scale_PC1 = 1.0/(X_pca[:,0].max() - X_pca[:,0].min())
    scale_PC2 = 1.0/(X_pca[:,1].max() - X_pca[:,1].min())
    
    #Extract the top three highest and lowest for each principle component
    loadings_copy = loadings.copy()
    three_highest_pc1 = loadings_copy.sort_values(by="PC1", ascending=False)[0:3]
    three_lowest_pc1 = loadings_copy.sort_values(by="PC1", ascending=True)[0:3]
    three_highest_pc2 = loadings_copy.sort_values(by="PC2", ascending=False)[0:3]
    three_lowest_pc2 = loadings_copy.sort_values(by="PC2", ascending=True)[0:3]

    loadings_copy = pd.concat([three_highest_pc1, three_lowest_pc1, three_highest_pc2, three_lowest_pc2])
    loadings_copy.to_excel("./sklearn_results/Figures/bleeders_PCA-loadings_top12.xlsx")
    ################################
    
    ###########
    #Plot the PCA of the bleeders and the loadings
    colors = ["#440154" if val==1 else '#5ec962' if val==2  else '#f98e09' for val in clusters]

    fig, ax = plt.subplots(figsize=(9,8))

    ax.scatter(X_pca[:,0]*scale_PC1, X_pca[:,1]*scale_PC2,color=colors)

    pc1_explained_variance, pc2_explained_variance = pca_obj.explained_variance_ratio_

    text_coordinates = [
        *[(0.35, 0)]*3,
        *[(-0.6, -0.05)]*3,
        *[(0, 0.4)]*3,
        *[(0, -0.3)]*3
    ]

    k = 0
    for i, (feature, text_coord) in enumerate(zip(loadings_copy.index, text_coordinates)):
        if k == 3:
            k = 0
        ax.arrow(0, 0, loadings_copy.iloc[i, 0], 
                loadings_copy.iloc[i, 1],
                head_width=0.02, 
                head_length=0.02,
                alpha=0.5,
                color="black")
        
        ax.text(text_coord[0], 
            text_coord[1]-0.05*k,
            feature, color="red", fontsize=12.5, alpha=0.8)
        
        k += 1
        
    ax.set_xlabel(f"PC1 ({pc1_explained_variance*100:.2f}%)", fontdict={"size":13})
    ax.set_ylabel(f"PC2 ({pc2_explained_variance*100:.2}%)", fontdict={"size":13})
    ax.set_xlim(-0.65, 0.8)

    fig.savefig(f"./sklearn_results/Figures/PCA_of_bleeders_with_loadings.pdf",  bbox_inches='tight')  
    ########################
    
    
    
if __name__=="__main__":
    if sys.argv[1] == "experiment":
        main(sys.argv[2])
    elif sys.argv[1] == 'calc_grid_space':
        count_length_of_grid_search()
    elif sys.argv[1] == 'generate_pictures_metrics':
        generate_metric_pictures()
    elif sys.argv[1] == 'unsupervised':
        perform_unsupervised_learning()


