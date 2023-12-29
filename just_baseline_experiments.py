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
from data_preparation import get_formatted_Baseline_FUP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from clinical_score import ClinicalScore
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, average_precision_score, roc_auc_score, recall_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector


import pandas as pd
import numpy as np
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

import re
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.feature_selection import chi2
from scipy.stats import ttest_ind
from statsmodels.stats import multitest
from sksurv.compare import compare_survival
import json
from sklearn.calibration import CalibratedClassifierCV
import joblib
from statistical_tests import *


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
    #Set the size of the fonts
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.title_fontsize'] = 13
    
    #Creates the graphs showing metrics average in 10-fold-nested cv
    def create_graph(ml_data_df, metric_name):
        fig, ax = plt.subplots(figsize=(10.5,5))
        
        ml_data = ml_data_df.copy()
        
        ml_data.loc[:,"model"] = ml_data["model"].apply(lambda x:nice_names[x])

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
            std = ml_data[ml_data['model']==ml_model]["value"].std()
            max_value =  ax.get_ylim()[1]
            ax.text(x_axis-0.35, max_value+(0.02*max_value), s=f" {mean:.2f}\n({std:.2f})", size=9)

        # ax.spines["right"].set_visible(False)
        # ax.spines["top"].set_visible(False)
        
        ax.tick_params(axis='x', labelrotation=45)
        #ax.set_title(metric_name, pad=30, fontsize=18)
        ax.set_xlabel("", fontsize=14)
        ax.set_ylabel(f"{metric_name}", fontsize=15)
        
        xlabel_to_type_dic = {v:all_models[k] for k,v in nice_names.items() if k in all_models}
        
        for i, model in enumerate(ordered_xlabels):
            if xlabel_to_type_dic[model]=="ML":
                plt.setp(ax.get_xticklabels()[i], color='red')

        
        fig.savefig(f"./sklearn_models/figures/{metric_name}.pdf", bbox_inches="tight")
        fig.savefig(f"./sklearn_models/figures/{metric_name}.png", dpi=500, bbox_inches="tight")

    #Populate the metrics data into a dic
    all_model_metrics = dict()

    #The metrics of interests
    metrics = ["AUPRC", "AUROC", "Brier Loss"]

    #Calculate and save the AUROC, AUPRC, and Brier score from the saved JSON files.
    all_model_metrics = get_CV_results_from_json(saving_path="./sklearn_models/test_results/")
        
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

    for model_name in [model for model in all_models if all_models[model]=="ML" if model != "Dummy"]:
            
        model = nice_names[model_name]
        
        feature_selection_counter[model] = {"PCA-5":0,"PCA-10":0, "SFS-5":0,"SFS-10":0, "None":0}
        
        for fold in [f"Fold_{i}" for i in range(1, 6)]:
            est1 = joblib.load(f'./sklearn_models/calibrated_models/{model_name}_{fold}_calibrated.pkl')
            feature_selection_method = est1.best_estimator_.steps[1][1]
            reduce_dim_method = est1.best_estimator_.steps[2][1]
        
            if feature_selection_method == reduce_dim_method: #Means both are passthrough, and no reduction in # of features occured
                feature_selection_counter[model]["None"] += 1
            elif reduce_dim_method != "passthrough": #Means PCA has occured
                if len(reduce_dim_method.components_) == 5:
                    feature_selection_counter[model]["PCA-5"] += 1
                elif len(reduce_dim_method.components_) == 10:
                    feature_selection_counter[model]["PCA-10"] += 1
            else:#Means Select from model occured
                if len(feature_selection_method.get_feature_names_out()) == 5:
                    feature_selection_counter[model]["SFS-5"] += 1
                elif len(feature_selection_method.get_feature_names_out()) == 10:
                    feature_selection_counter[model]["SFS-10"] += 1
            
    fig, ax = plt.subplots(figsize=(8/1.5, 4/1.5))
    

    feat_select_df = pd.DataFrame.from_dict(feature_selection_counter).T
    # feat_select_df["model"] = feat_select_df.index
    # feat_select_df = feat_select_df.melt(id_vars ="model", value_vars=["PCA", "SFS", "None"], var_name="method")

    # sns.barplot(data=feat_select_df, x="model", y="value", hue="method", ax=ax, palette="icefire", width=0.6)
    feat_select_df.plot(kind="bar", stacked=True, ax=ax, color=["#d15e56", "#d13328",
                                                            "#ba64d1", "#b034d1",
                                                            "#B0A8B9"])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    
    ax.xaxis.set_tick_params(labelsize=9, rotation=0)
    ax.yaxis.set_tick_params(labelsize=9)

    fig.savefig(f"./sklearn_models/figures/best_selection_methods.pdf", bbox_inches="tight")
    fig.savefig(f"./sklearn_models/figures/best_selection_methods.png", dpi=300, bbox_inches="tight")
            

def perform_statistical_tests():
    """Perform pairwise Wilcoxon sign-rank test and show their results as a heatmap
    """
    metric_names = ["AUPRC", "AUROC", "Brier Loss"]
    modes = ["all_pairs","ML vs Clinical"]

    grid_search_results_path = "./sklearn_models/test_results/"
    stat_figure_save_path = "./sklearn_models/figures/"

    for mode in modes:
        for metric_name in metric_names:

            all_model_metrics = get_CV_results_from_json(grid_search_results_path)

            omnibus_results = omnibus_test(all_model_metrics, metric_name, method="Friedman")

            effect_df = calc_effect_size(all_models, all_model_metrics, metric_name=metric_name, mode=mode)    

            stat_df = calc_pairwise_p_value(all_models, all_model_metrics, metric_name=metric_name, method="Wilcoxon signed-rank test", mode=mode)

            stat_df_corrected, multitest_used = correct_p_values(stat_df, multitest_correction="fdr_bh")

            plot_p_value_heatmap(stat_df_corrected, effect_size_df=effect_df, title=metric_name, 
                                save_path = stat_figure_save_path,
                                multitest_correction = multitest_used,
                                plot_name=f"{metric_name}_{mode}_", 
                                omnibus_p_value=f"{omnibus_results}", 
                                p_value_threshold=0.05)
            

def get_param_grid_model(classifier, joblib_memory_path = None):
    
    #Params for sequential feature selection
    sfs_scoring = "roc_auc"
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
        
        model = SVC(random_state=1, probability=True)
        
        
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
        item['preprocess'] = [StandardScaler(), MinMaxScaler()]

                                      
        
    pipe = Pipeline([('preprocess', 'passthrough'),
                     ('feature_selection', 'passthrough'),
                    ('reduce_dim','passthrough'),
                    ('model', model)],
                    memory=joblib_memory_path)
    
    return pipe, param_grid


def create_5_fold_cv_patient_ids():
    """Saves the patient ids as 80% training-val, 10% calibration, 10% testing for 5 folds as JSON file in a stratified fashion.
        This is done for reprodicibility accross multiple platforms.
    """
    #Get the formatted baseline dataset
    _, _, _, concat_x, target_series = get_formatted_Baseline_FUP(mode="Formatted")

    #Perform nested cross validation
    #1 get outer cv indexes
    from sklearn.model_selection import StratifiedKFold

    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    fold_id_dict = {f"Fold_{fold}":{"train_val":[], "test":[], "calibration":[]} for fold in [1,2,3,4,5]}

    #Extract the train-val (80% or 5-fold split), and calibration_test (20%)
    for i, (train_index, caliber_test_index) in enumerate(outer_cv.split(concat_x, target_series)):
        
        #Getting the patients ids for train_val
        fold_id_dict[f"Fold_{i+1}"]["train_val"] = list(target_series.iloc[train_index].index)
        
        #Get ids for caliber and test
        X_caliber_test = concat_x.iloc[caliber_test_index]
        y_caliber_test = target_series.iloc[caliber_test_index]
        
        #Get the ids
        for j, (caliber_index, test_index) in enumerate(inner_cv.split(X_caliber_test, y_caliber_test)):
            
            fold_id_dict[f"Fold_{i+1}"]["calibration"] = list(y_caliber_test.iloc[caliber_index].index)
            fold_id_dict[f"Fold_{i+1}"]["test"] = list(y_caliber_test.iloc[test_index].index)

    for fold in fold_id_dict:
        
        print(f"Fold {fold}") 
        for data_section in ["train_val", "test", "calibration"]:
            data_section_ids = fold_id_dict[fold][data_section]
            x = concat_x.loc[data_section_ids]
            y = target_series.loc[data_section_ids]
            

            print("\t",f" {data_section} bleeders:{sum(y)} non-bleeders:{len(y)-sum(y)}")
            
    with open('5_fold_cv_ids.json', 'w', encoding='utf-8') as f: 
        json.dump(fold_id_dict, f, ensure_ascii=False, indent=4)
        

def perform_nested_cv_with_calibration(model, X, y, joblib_memory_path = None):
    """Perform 5-fold nested CV by training and optimizing the ML models on the training-val fold, calibrating on the calibration fold, and then testing on the testing fold.
    Note: Requires '5_fold_cv_ids.json' file with patient ids for 5 fold cv.

    Args:
        model (str): Name of the ML or clinical model.
        X (pd.DataFrame): Tabular dataframe of all the baseline dataset.
        y (pd.Series): Series of the targets for bleeders and non-bleeders.
        joblib_memory_path (str, optional): If ML grid search should be stored for efficiency. Defaults to None.
    """
    #Get the dictionary with the cross-validation ids
    with open('5_fold_cv_ids.json', 'r', encoding='utf-8') as f: 
        fold_id_dict = json.load(f)
    
    
    #Stratification object for hyperparameter optimization
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    print(f'Started performing nested cv for {model}')
            
    for fold_num in fold_id_dict:
        
        start_time = time.time()
        
        #Copy the input X and the output y  
        baseline_dataframe_duplicate = X.copy()
        target_series_duplicate = y.copy()
        
        train_val_ids = fold_id_dict[fold_num]["train_val"]
        test_ids = fold_id_dict[fold_num]["test"]
        calibration_ids = fold_id_dict[fold_num]["calibration"]
        
        #Perform hyperparameter optimization followed by data training on the training_val dataset
        x_training_val = baseline_dataframe_duplicate.loc[train_val_ids]
        y_training_val = target_series_duplicate.loc[train_val_ids]
        
        
        if all_models[model] == "ML":
            #Get the parameter grid
            pipe, param_grid = get_param_grid_model(model, joblib_memory_path)
            
            clf = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1, refit=True)
            #clf = RandomizedSearchCV(estimator=pipe, n_iter=10, param_distributions=param_grid, cv=inner_cv, scoring="roc_auc",n_jobs=-1, refit=True)
            
            #Fit the model and do hyperparmeter optimization
            clf.fit(x_training_val, y_training_val)
        
        elif all_models[model] == "Clinical":
            #Create the score object
            clf = ClinicalScore(model)

            #Fitting doesn't do anything special, it is just required for the class
            clf = clf.fit(x_training_val, y_training_val)
        
    
        #Calibrate the fitted model
        x_calibration = baseline_dataframe_duplicate.loc[calibration_ids]
        y_calibration = target_series_duplicate.loc[calibration_ids]
        
        calibrated_clf = CalibratedClassifierCV(clf, cv="prefit")
        calibrated_clf.fit(x_calibration, y_calibration)
        
        
        #Test the calibrated, fitted model
        x_test = baseline_dataframe_duplicate.loc[test_ids]
        y_test = target_series_duplicate.loc[test_ids]
        #save the calibrated and uncalibrated models
        joblib.dump(clf, f"./sklearn_models/uncalibrated_models/{model}_{fold_num}_NOT_calibrated.pkl")
        joblib.dump(clf, f"./sklearn_models/calibrated_models/{model}_{fold_num}_calibrated.pkl")
        #Test results
        test_results = {
            "uniqid": list(x_test.index),
            "y_actual": list(y_test),
            "y_pred_NOT_calibrated": list(clf.predict_proba(x_test)[:, 1]),
            "y_pred_calibrated": list(calibrated_clf.predict_proba(x_test)[:, 1])
        }
        #Save the detailed test results
        with open(f'./sklearn_models/test_results/{model}_{fold_num}_detailed_test_results.json', 'w', encoding='utf-8') as f: 
            json.dump(test_results, f, ensure_ascii=False, indent=4, default=int)
        
        
        print("\t", model, fold_num, "is done in", f'{time.time()-start_time}', "seconds.")    


def perform_nested_cv(model, X, y, joblib_memory_path = None):
    
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
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    
    print(f'Started performing nested cv for {model}')
    start_time = time.time()
    
    if all_models[model] == "ML":
        
        pipe, param_grid = get_param_grid_model(model, joblib_memory_path)



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


def main(mode, joblib_memory_path=None):
    
    def nested_cv_(model_name):
        if all_models[model_name] == "Clinical":
            _, _, _, concat_x, target_series = get_formatted_Baseline_FUP(mode="Raw")
            perform_nested_cv_with_calibration(model_name, concat_x, target_series, joblib_memory_path)
        elif all_models[model_name] == "ML":
            _, _, _, concat_x, target_series = get_formatted_Baseline_FUP(mode="Formatted")
            perform_nested_cv_with_calibration(model_name, concat_x, target_series, joblib_memory_path)
    
    
    #Perform nested cv
    if mode == "all_models":
        for model in all_models:
            nested_cv_(model)
    elif mode in all_models.keys():
        nested_cv_(mode)
    elif mode == "test":
        _, _, _, concat_x, target_series = get_formatted_Baseline_FUP(mode="Formatted")

        print("Data preparation was finished succesfully.")
    else:
        raise Exception(f"The mode {mode} doesn't exist.")
    

def perform_unsupervised_learning():
    
    #Get the Baseline dataset formatted
    patient_dataset, _, _, concat_x, target_series = get_formatted_Baseline_FUP(mode="Formatted")

     
    ###########################
    #Calculate the time since bleeding from baseline and categorize patients
    
    bleeding_since_baseline_category = []
    bleeding_since_baseline_duration = []
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
                        bleeding_since_baseline_category.append(year_since_bleed_str)
                        bleeding_since_baseline_duration.append(year_since_bleed)
                        
            #There are patients who have the AD1 filled for them, but they didn't bleed
            if number_of_maj_bleed == 0:
                bleeding_since_baseline_category.append("Non-bleeders")

        else:
            bleeding_since_baseline_category.append("Non-bleeders")
            bleeding_since_baseline_duration.append(-1)
            
    ###################################
    
    
    #Scale the baseline dataset for unsupervised learning
    scaler = MinMaxScaler()
    concat_scaled_df = pd.DataFrame(scaler.fit_transform(concat_x),  columns = concat_x.columns)
    
    ###########################
    #Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(15*1.5, 12*1.5))

    corr_df = concat_scaled_df.corr("pearson")
    mask_matrix = np.triu(np.ones_like(corr_df))

    sns.heatmap(corr_df, cmap="vlag", linewidth=.5, mask=mask_matrix, cbar_kws={"shrink":0.4, "pad":0, "anchor":(-1, 0.8), "label":"Pearson Correlation"})
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 10)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 10)
    
    fig.savefig(f"./sklearn_results/Figures/correlation_matrix.png", dpi=500, bbox_inches='tight')  
    
    corr_list = []
    for i in range(len(corr_df)):
        for j in range(i+1):
            if i != j:
                if np.abs(corr_df.iloc[i, j]) >0.5:
                    corr_list.append({"Feature 1":corr_df.columns[i], "Feature 2":corr_df.columns[j], "Pearson Correlation":corr_df.iloc[i, j]})

    corr_list = pd.DataFrame.from_dict(corr_list).sort_values(by="Pearson Correlation", ascending=False)
    corr_list.to_csv("./sklearn_results/Figures/moreThan50_correlation.csv")
    #############################
    
    #Dimentionality reduction techniques
    feature_decomp_method_dics = {
        "Kernel PCA": KernelPCA(n_components=2, kernel="rbf",  gamma=1/5, fit_inverse_transform=False),
        "Isomap": Isomap(n_components=2, n_neighbors=20, p=1),
        "t-SNE": TSNE(n_components=2, perplexity=5, init="random", n_iter=250, random_state=0),    
    }

    # y = ["bleeders" if val==1 else "non-bleeders" for val in target_series ]
    # colors = ["red" if val==1 else "blue" for val in target_series ]
    
    #######################################
    #Figure with lower dimention visualized

    colors = ["#ed6925" if status != "Non-bleeders" else "#000004" for status in bleeding_since_baseline_category]
    alphas = [1 if status != "Non-bleeders" else 0.5 for status in bleeding_since_baseline_category]
    zorders = [3 if status != "Non-bleeders" else -1 for status in bleeding_since_baseline_category ]
    markers = ["o" if status != "Non-bleeders" else "o" for status in bleeding_since_baseline_category ]
    edgecolors = [None if status != "Non-bleeders" else None for status in bleeding_since_baseline_category ]
    labels = ["Bleeders" if status != "Non-bleeders" else "Non-bleeders" for status in bleeding_since_baseline_category ]

    def legend_without_duplicate_labels(ax):
        ax.legend( fontsize = 6)

        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='upper center', bbox_to_anchor=(0.5, -0.05),
            ncol=2, fontsize=12)

    #########################################
    #Create the PCA and explained variance graph        
    fig, axs = plt.subplots(1, 2, figsize=(8.3, 5.6/2), layout="constrained", gridspec_kw={"wspace":0.15})

    #Draw PCA
    ax = axs.flat[0]
    pca_obj = PCA()
    pca_obj = pca_obj.fit(concat_scaled_df)
    X_processed = pca_obj.transform(concat_scaled_df)

    for i, (color, alpha, zorder, label, marker, edgecolor) in enumerate(zip(colors, alphas, zorders, labels, markers, edgecolors)):
        ax.scatter(X_processed[i,0], X_processed[i,1], marker=marker, c = color, s=12, linewidth=0.1, label=label, zorder= zorder, alpha=alpha)

    # ax.get_yaxis().set_visible(False)
    # ax.get_xaxis().set_visible(False)
    ax.set_xlabel(f"PC1 ({pca_obj.explained_variance_ratio_[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pca_obj.explained_variance_ratio_[1]*100:.2f}%)")

    ax.set_title("PCA", fontsize=15)

    #Draw PCA explained variance
    ax = axs.flat[1]
    cumsum_explained_variance = np.cumsum(pca_obj.explained_variance_ratio_)
    ax.plot( range(1, len(cumsum_explained_variance) +1), cumsum_explained_variance, "-o", color="black")
    percent95_explained_var_index = np.argmax(cumsum_explained_variance>0.95)+1
    ax.hlines(0.95, -1, percent95_explained_var_index, linestyles="dotted", color = "grey")
    ax.vlines(percent95_explained_var_index, -1, 0.95, linestyles="dotted", color = "grey")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Percentage of Variance")
    # ax.text(percent95_explained_var_index+2, 0.1, f"x = {percent95_explained_var_index}")
    
    fig.savefig(f"./sklearn_results/Figures/PCA_Explained_varience_all_new.pdf",  bbox_inches='tight')  
    fig.savefig(f"./sklearn_results/Figures/PCA_Explained_varience_all_new.png", dpi=500, bbox_inches='tight')  

    
    #################################################
    #Create the graph of three dimentionality reduction
    
    fig, axs = plt.subplots(1, 3, figsize=(8.3, 5.6/2), layout="constrained", gridspec_kw={"wspace":0.1})


    for method, ax in zip(feature_decomp_method_dics, axs.flat):
        
        X_processed = feature_decomp_method_dics[method].fit_transform(concat_scaled_df)
        

        for i, (color, alpha, zorder, label, marker, edgecolor) in enumerate(zip(colors, alphas, zorders, labels, markers, edgecolors)):
            ax.scatter(X_processed[i,0], X_processed[i,1], marker=marker, c = color, s=12, linewidth=0.1, label=label, zorder= zorder, alpha=alpha)

        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        ax.set_title(method, fontsize=15)

    legend_without_duplicate_labels(axs.flat[1])

    fig.savefig(f"./sklearn_results/Figures/dimentionality_reduction_all_new.pdf",  bbox_inches='tight')  
    fig.savefig(f"./sklearn_results/Figures/dimentionality_reduction_all_new.png", dpi=500, bbox_inches='tight')  

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


    alphas = [1 if status != "Non-bleeders" else 1 for status in bleeding_since_baseline_category]
    zorders = [3 if status != "Non-bleeders" else -1 for status in bleeding_since_baseline_category ]
    markers = ["X" if status != "Non-bleeders" else "o" for status in bleeding_since_baseline_category ]
    point_size = [40 if status != "Non-bleeders" else 14 for status in bleeding_since_baseline_category ]
    edgecolors = ["yellow" if status != "Non-bleeders" else None for status in bleeding_since_baseline_category ]


    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'hspace': 0.15, 'wspace': 0.05})

    for method, ax in zip(clustering_methods_dic, axs.flat):



        clusters = clustering_methods_dic[method].fit(concat_scaled_df)
        
        colors = ["#f98e09" if label != 0 else "#57106e" for label in clusters.labels_]


        for i, (color, alpha, zorder, label, marker, edgecolor, s_size) in enumerate(zip(colors, alphas, zorders, bleeding_since_baseline_category, markers, edgecolors, point_size)):
            ax.scatter(X_PCA[i,0], X_PCA[i,1], marker=marker, c = color, s=s_size, linewidth=0.3, label=label, zorder= zorder, 
                    alpha=alpha, edgecolors=edgecolor)

        # legend_without_duplicate_labels(ax)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        ax.text(0.60, 0.99, ha='left', va='top',transform=ax.transAxes,size=11,
                s=f"Homogeneity: {homogeneity_score(target_series ,clusters.labels_):.2e} \nCompleteness: {completeness_score(target_series,clusters.labels_):.2e}")

        ax.set_title(method, fontsize=16)
        
    axs[1].legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.04, 0.5), fontsize=12)
        
    fig.savefig(f"./sklearn_results/Figures/clustering_pic_all_points_new.pdf",  bbox_inches='tight')  
    fig.savefig(f"./sklearn_results/Figures/clustering_pic_all_points_new.png", dpi=500,  bbox_inches='tight')  
    ###########################################
    
    #######################################
    #Clustering and PCA just on the bleeders
    #Just analyse the bleeders now
    only_bleeders_X = concat_x[target_series==1].copy()
    only_bleeders_X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(only_bleeders_X), columns=only_bleeders_X.columns, index=only_bleeders_X.index)    
    
    #########
    #The agglomerative clustering on the bleeders
    fig, ax = plt.subplots(figsize=(15,5))

    Z = linkage(only_bleeders_X_scaled, 'ward')
    
    bleeders_cluster_color_pallette = ["#440154",'#5ec962', '#f98e09']

    hierarchy.set_link_color_palette(bleeders_cluster_color_pallette)

    max_d = 9

    a = dendrogram(Z, labels=only_bleeders_X_scaled.index, leaf_rotation=90, ax=ax, color_threshold=max_d)

    ax.axhline(y=max_d, xmin = 0, xmax=1000, color='gray', linestyle="dotted")

    ax.get_xaxis().set_visible(False)

    ax.set_ylabel("Distance", fontsize=20)
    ax.yaxis.set_tick_params(labelsize=15)

    # ax.text(x=10, y=30, s=f"Cutoff: {max_d}", size=13)

    fig.savefig(f"./sklearn_results/Figures/agglomerative_clustering_bleeders_new.pdf",  bbox_inches='tight')  
    fig.savefig(f"./sklearn_results/Figures/agglomerative_clustering_bleeders_new.png", dpi=500,  bbox_inches='tight')  
    ##########
    
    ####################################################
    #Create the Kaplan Meier plot
    #Clusters as defined by the agglomerative clustering
    clusters = fcluster(Z, max_d, criterion='distance')
    
    #Print number of patients in each cluster
    print([f"Cluster {cluster_num}: {len(clusters[clusters==cluster_num])}" for cluster_num in set(clusters)])
    
    only_bleeders_X["cluster"] = clusters
    #Time in months
    only_bleeders_X["time_to_bleed_months"] = [val*12 for val in bleeding_since_baseline_duration if val != -1]

    #Plot the Kaplan Meier curve
    fig, ax = plt.subplots(figsize=(4, 2.5))
    for group, color in zip(sorted(set(clusters)), bleeders_cluster_color_pallette):
        
        data_subset = only_bleeders_X[only_bleeders_X["cluster"]==group].copy()

        time, survival_prob, conf_int = kaplan_meier_estimator(
            [True]*len(data_subset), data_subset["time_to_bleed_months"], conf_type="log-log"
        )
        plt.step(time, 1-survival_prob, where="post", color=color, label = f"Cluster {group}")
        # plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.legend(loc="lower right")
    plt.ylim(0, 1)
    plt.ylabel("Probability of Bleeding")
    plt.xlabel("Months")

    fig.savefig(f"./sklearn_results/Figures/Kaplan_Meier.pdf",  bbox_inches='tight')  
    fig.savefig(f"./sklearn_results/Figures/Kaplan_Meier.png", dpi=500,  bbox_inches='tight')  
    
    ###################
    #Perform log-rank test to see if the two clusters are statistically significantly different
    arr_content = [(True, val) for val in only_bleeders_X["time_to_bleed_months"]]
    arr_content = np.array(arr_content, dtype=np.dtype([('bool', bool), ('val', float)]))
    chisq, p_val, stats, covar_mat =  compare_survival(y=arr_content  ,group_indicator=only_bleeders_X["cluster"], return_stats=True)
    print(f"\nThe log-rank p-value is {p_val}\n")
    print(stats, "\n")
    #################
    #Perform one Cluster vs rest to find the discriminaotory features
    # fig, axs = plt.subplots(1, 3, figsize=(15, 3.5), constrained_layout=True)

    # for ax, cluster_num, color in zip(axs.flat, set(clusters), bleeders_cluster_color_pallette):

    #     x_cluster = only_bleeders_X.copy()
    #     x_cluster = x_cluster.drop(["cluster","time_to_bleed_months"], axis=1)
    #     x_cluster = pd.DataFrame(StandardScaler().fit_transform(x_cluster), columns=x_cluster.columns)
    #     y_cluster = only_bleeders_X["cluster"].copy()
    #     y_cluster = [1 if val == cluster_num else 0 for val in y_cluster]
        
        
    #     reg = LassoCV(cv=3, random_state=0, max_iter=5000).fit(x_cluster, y_cluster)

    #     coef_df = pd.DataFrame({"Feature Name":reg.feature_names_in_, "coef":reg.coef_})
    #     coef_df = coef_df[np.abs(coef_df["coef"]) > 0]
    #     coef_df = coef_df.sort_values(by="coef", ascending=True)
    #     coef_df = pd.concat([coef_df[0:5],coef_df[-5:] ])
    #     ax.barh(coef_df["Feature Name"], coef_df["coef"], color=color)
    #     ax.set_title(f"Cluster {cluster_num}")
    #     ax.set_xlabel("Coefficient")
        
    # fig.savefig(f"./sklearn_results/Figures/LasssoCV_results.pdf",  bbox_inches='tight')  
    # fig.savefig(f"./sklearn_results/Figures/LasssoCV_results.png", dpi=500,  bbox_inches='tight')  
    ###################
    #Lasso on the two clusters
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    x_cluster = only_bleeders_X.copy()
    x_cluster = x_cluster.drop(["cluster","time_to_bleed_months"], axis=1)
    x_cluster = pd.DataFrame(StandardScaler().fit_transform(x_cluster), columns=x_cluster.columns)
    y_cluster = only_bleeders_X["cluster"].copy()
    y_cluster = [1 if val == 1 else 0 for val in y_cluster]


    reg = LogisticRegressionCV(cv=2, random_state=0, max_iter=5000, penalty='l1', solver="liblinear", Cs=[0.01, 0.05, 0.1, 0.15]).fit(x_cluster, y_cluster)

    coef_df = pd.DataFrame({"Feature Name":reg.feature_names_in_, "coef":reg.coef_[0]})
    coef_df = coef_df[coef_df["coef"]!=0].sort_values(by="coef")
    
    ########################
    #Perform p_value calculations of the features between the two clusters
    coef_df = coef_df.sort_values(by="coef", ascending=False)

    percentage_df = []
    for feat in coef_df["Feature Name"]:
        #If the column was not categorical we do mean +- std and perform t-test to see their difference
        if len(set(only_bleeders_X[feat])) > 2:
            mean_cluster_1 = only_bleeders_X.loc[only_bleeders_X["cluster"] == 1, feat].mean()
            std_cluster_1 = only_bleeders_X.loc[only_bleeders_X["cluster"] == 1, feat].std()
            mean_cluster_2 = only_bleeders_X.loc[only_bleeders_X["cluster"] == 2, feat].mean()
            std_cluster_2 = only_bleeders_X.loc[only_bleeders_X["cluster"] == 2, feat].std()
    
            _ ,p_val  = ttest_ind(only_bleeders_X.loc[only_bleeders_X["cluster"] == 1, feat], only_bleeders_X.loc[only_bleeders_X["cluster"] == 2, feat])
            
            percentage_df.append({"Feature Name": feat, "Cluster 1":f"{mean_cluster_1:.2f}±{std_cluster_1:.2f}" , "Cluster 2":f"{mean_cluster_2:.2f}±{std_cluster_2:.2f}", "p-value": p_val})
        
        #If the column was not categorical we do % prevalence and perform chi2 to see their difference
        else:
            percent_cluster_1 = only_bleeders_X.loc[only_bleeders_X["cluster"] == 1, feat].sum()/len(only_bleeders_X.loc[only_bleeders_X["cluster"] == 1, feat])*100
            percent_cluster_2 = only_bleeders_X.loc[only_bleeders_X["cluster"] == 2, feat].sum()/len(only_bleeders_X.loc[only_bleeders_X["cluster"] == 2, feat])*100
            
            _ ,p_val  = chi2(only_bleeders_X[feat].to_frame(), only_bleeders_X["cluster"])
            
            p_val = list(p_val)[0]        
            
            percentage_df.append({"Feature Name": feat, "Cluster 1":f"{percent_cluster_1:.2f}%" , "Cluster 2":f"{percent_cluster_2:.2f}%", "p-value": p_val})
    
    #Create a df with the column values and p-values        
    percentage_df = pd.DataFrame.from_dict(percentage_df)
    percentage_df = percentage_df.set_index("Feature Name")

    #Multihypothesis correction with bh method
    _, corrected_pvalues, _, _ = multitest.multipletests(percentage_df["p-value"], alpha=0.05, method="fdr_bh")
    percentage_df["p-value"] = [f"{val:.2e}" for val in corrected_pvalues]
    
    percentage_df.to_csv("./sklearn_results/Figures/p_values_of_features_between_two_clusters.csv")

    ################################

    coef_df["Feature Name"] = coef_df["Feature Name"].apply(lambda x: x+" *" if float(percentage_df.loc[x,"p-value"])<0.05 else x)

    coef_df = coef_df.sort_values(by="coef")
    
    ax.barh(coef_df["Feature Name"], coef_df["coef"], color=bleeders_cluster_color_pallette[2])
    ax.set_xlabel("Coefficient", size=12)
    ax.yaxis.set_tick_params(size=12)
    
    fig.savefig(f"./sklearn_results/Figures/LasssoCV_results.pdf",  bbox_inches='tight')  
    fig.savefig(f"./sklearn_results/Figures/LasssoCV_results.png", dpi=500,  bbox_inches='tight')  
  
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
    loadings.to_excel("./sklearn_results/Figures/bleeders_PCA-loadings_all_new.xlsx")

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
    loadings_copy.to_excel("./sklearn_results/Figures/bleeders_PCA-loadings_top12_new.xlsx")
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

    fig.savefig(f"./sklearn_results/Figures/PCA_of_bleeders_with_loadings_new.pdf",  bbox_inches='tight')  
    fig.savefig(f"./sklearn_results/Figures/PCA_of_bleeders_with_loadings_new.png", dpi=500, bbox_inches='tight')  
    ########################
    
    
def get_CV_results_from_json(saving_path):
    """Read the JSON summary files of 5-fold nested CV stored in saving_path, then calculates AUROC, AUPRC, and Brier Loss for each fold of CV.

    Args:
        saving_path (str): Path to the folder containing the json files.

    Returns:
        dict: A dictionary where keys are model names and values are the metrics measured for 5-fold cv.
    """
    #Populate the metrics data into a dic
    all_model_metrics = dict()

    #Populate the dic
    for model_name in all_models.keys():

        algorithm_metric_dicts = {"AUROC":[], "AUPRC":[], "Brier Loss":[]}
        
        for fold in [f"Fold_{i}" for i in range(1, 6)]:
            
            with open(f'{saving_path}/{model_name}_{fold}_detailed_test_results.json', 'r', encoding='utf-8') as f: 
                fold_id_dict = json.load(f)
            
            y_actual = fold_id_dict["y_actual"]
            y_pred_calibrated = fold_id_dict["y_pred_calibrated"]
            
            
            auroc = roc_auc_score(y_actual, y_pred_calibrated)
            auprc = average_precision_score(y_actual, y_pred_calibrated)
            brier = brier_score_loss(y_actual, y_pred_calibrated)
            
            algorithm_metric_dicts["AUROC"].append(auroc)
            algorithm_metric_dicts["AUPRC"].append(auprc)
            algorithm_metric_dicts["Brier Loss"].append(brier)
        
        
        all_model_metrics[model_name] = algorithm_metric_dicts
        
    return all_model_metrics
    
if __name__=="__main__":
    """
    - To perform supervised analysis:
        Option 1 (no Temp directory to save Scikit-learn models): 
            python just_baseline_experiments.py experiment {all_models; MODEL_NAME}
        Option 2 (with temp directory)
            python just_baseline_experiments.py experiment {all_models; MODEL_NAME} TMP_directory
    """

    if (sys.argv[1] == "experiment"):
        if (len(sys.argv) == 4):
            print(f"The temp directory is set to: {sys.argv[3]}")
            main(sys.argv[2], sys.argv[3])
        else:
            print("No temp directory will be used.")
            main(sys.argv[2])
    elif sys.argv[1] == 'calc_grid_space':
        count_length_of_grid_search()
    elif sys.argv[1] == 'generate_pictures_metrics':
        generate_metric_pictures()
    elif sys.argv[1] == 'unsupervised':
        perform_unsupervised_learning()
    elif sys.argv[1] == 'statistical_tests':
        perform_statistical_tests()



