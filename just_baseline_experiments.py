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
import seaborn as sns



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
    
    def create_graph(ml_data_df, metric_name):
        fig, ax = plt.subplots(figsize=(10.5,5))
        
        ml_data = ml_data_df.copy()
        
        ml_data.loc[:,"model"] = ml_data["model"].apply(lambda x:nice_names[x])

        #Set the size of the fonts
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.title_fontsize'] = 18

        #Order the MLs based on the mean of their score
        order = {ml_model:ml_data[ml_data['model']==ml_model]["value"].mean() for ml_model in ml_data['model'].unique()}
        ordered_xlabels = [k for k, v in sorted(order.items(),reverse=True, key=lambda item: item[1])]

        sns.boxplot(data=ml_data, x="model", y="value", ax=ax, linewidth=1, width=0.4, order=ordered_xlabels)

        #Change the transparency of box plot
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .3))

        sns.swarmplot(data=ml_data, x="model", y="value", hue="test_set", ax=ax, size=5, palette="colorblind", order=ordered_xlabels)

        for x_axis, ml_model in enumerate(ordered_xlabels):
            mean = ml_data[ml_data['model']==ml_model]["value"].mean()
            max_value =  ax.get_ylim()[1]
            ax.text(x_axis-0.35, max_value+(0.02*max_value), s=f"μ={mean:.3f}", size=6)

        # ax.spines["right"].set_visible(False)
        # ax.spines["top"].set_visible(False)
        
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_title(metric_name, pad=30)
        ax.set_xlabel("")
        ax.get_legend().remove()
        
        xlabel_to_type_dic = {v:all_models[k] for k,v in nice_names.items()}
        
        for i, model in enumerate(ordered_xlabels):
            if xlabel_to_type_dic[model]=="ML":
                plt.setp(ax.get_xticklabels()[i], color='red')

        
        fig.savefig(f"./sklearn_results/Figures/{metric_name}.png", dpi=300, bbox_inches="tight")

    all_model_metrics = dict()

    metrics = ["Accuracy","Precision","Sensitivity", "AUPRC", "AUROC","F1-score", "Specificity","TN","TP","FN","FP"]


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
        
        
    df_all = pd.DataFrame()
    for model in all_model_metrics.keys():

        df = pd.DataFrame.from_dict(all_model_metrics[model])
        df["model"] = model
        df_all = pd.concat([df_all, df])

    number_cvs = int(len(df_all)/len(all_models))
    
    df_all["test_set"] = [f'test-split-{i+1}' for i in range(number_cvs)]*len(all_models)

    df_all = df_all.melt(id_vars=['model', "test_set"])

    for metric in metrics:
    
        data = df_all[df_all["variable"]==metric]
        create_graph(data, metric)
        


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
            }
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
    
    
if __name__=="__main__":
    if sys.argv[1] == "experiment":
        main(sys.argv[2])
    elif sys.argv[1] == 'calc_grid_space':
        count_length_of_grid_search()
    elif sys.argv[1] == 'generate_pictures_metrics':
        generate_metric_pictures()


