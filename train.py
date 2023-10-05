#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""The main module to run the experiments.

Usage:
    python train.py [Option]

    Option:"Baseline_Dense", "LastFUP_Dense", "FUP_RNN", "FUP_Baseline_Multiinput", "Dummy_classifiers"

"""
# =============================================================================
# Imports
from hypermodel_experiments import run_lastFUP_dense_experiment, run_baseline_dense_experiment, run_FUP_RNN_experiment, run_Baseline_FUP_multiinput_experiment, run_dummy_experiment, run_ensemble_experiment
import sys
from data_preparation import get_formatted_Baseline_FUP

def main(experiment_name):
    
    patient_dataset, FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = get_formatted_Baseline_FUP(mode="Formatted")
    
    print(f"Follow-up data has {len(FUPS_dict)} examples and {len(list_FUP_cols)} features.")
    print(f"Baseline data has {len(baseline_dataframe)} examples and {len(baseline_dataframe.columns)} features.")
    print(f"The target data has {len(target_series)} data.", "\n")
    
    if experiment_name == "LastFUP_Dense":
        print(f"Performing {experiment_name} experiment. \n")
    
        run_lastFUP_dense_experiment(model_name = f"{experiment_name}", 
                                    directory_name = f"./keras_tuner_results/{experiment_name}", 
                                    metric_name = "prc", 
                                    metric_mode = "max", 
                                    metric_cv_calc_mode = "median", 
                                    baseline_dataframe = baseline_dataframe,
                                    FUPS_dict = FUPS_dict,
                                    target_series = target_series,
                                    list_FUP_cols = list_FUP_cols,
                                    patient_dataset = patient_dataset,
                                    overwrite=False
                                    )
        
    elif experiment_name == "Baseline_Dense":
        print(f"Performing {experiment_name} experiment. \n")

        run_baseline_dense_experiment(model_name = f"{experiment_name}", 
                                    directory_name = f"./keras_tuner_results/{experiment_name}", 
                                    metric_name = "prc", 
                                    metric_mode = "max", 
                                    metric_cv_calc_mode = "median", 
                                    baseline_dataframe = baseline_dataframe,
                                    FUPS_dict = FUPS_dict,
                                    target_series = target_series,
                                    list_FUP_cols = list_FUP_cols,
                                    patient_dataset = None,
                                    overwrite=False
                                    )

    elif experiment_name == "FUP_RNN":
        print(f"Performing {experiment_name} experiment. \n")

        run_FUP_RNN_experiment(model_name = f"{experiment_name}", 
                                    directory_name = f"./keras_tuner_results/{experiment_name}", 
                                    metric_name = "prc", 
                                    metric_mode = "max", 
                                    metric_cv_calc_mode = "median", 
                                    baseline_dataframe = baseline_dataframe,
                                    FUPS_dict = FUPS_dict,
                                    target_series = target_series,
                                    list_FUP_cols = list_FUP_cols,
                                    patient_dataset = patient_dataset,
                                    overwrite=False)
        
        
    elif experiment_name == "FUP_Baseline_Multiinput":
        print(f"Performing {experiment_name} experiment. \n")
        
        run_Baseline_FUP_multiinput_experiment(model_name = f"{experiment_name}", 
                                    directory_name = f"./keras_tuner_results/{experiment_name}", 
                                    metric_name = "prc", 
                                    metric_mode = "max", 
                                    metric_cv_calc_mode = "median", 
                                    baseline_dataframe = baseline_dataframe,
                                    FUPS_dict = FUPS_dict,
                                    target_series = target_series,
                                    list_FUP_cols = list_FUP_cols,
                                    patient_dataset = patient_dataset,
                                    overwrite=False)
        
    elif experiment_name == "Dummy_classifiers":
        print(f"Performing {experiment_name} experiment. \n")
        
        run_dummy_experiment(model_name = "Dummy_classifiers", 
                        baseline_dataframe = baseline_dataframe,
                        FUPS_dict = FUPS_dict,
                        target_series = target_series)
        
    elif experiment_name == "Ensemble":
        print(f"Performing {experiment_name} experiment. \n")
        
        run_ensemble_experiment(model_name= "Ensemble",
                                directory_name = f"./keras_tuner_results/{experiment_name}", 
                                baseline_dataframe = baseline_dataframe,
                                FUPS_dict = FUPS_dict,
                                target_series = target_series,
                                list_FUP_cols = list_FUP_cols,
                                patient_dataset = patient_dataset)
    
    else:
        raise ValueError(f"There is no experiment name called {experiment_name}")
    

    
if __name__=="__main__":
    main(experiment_name = sys.argv[1])
    
    