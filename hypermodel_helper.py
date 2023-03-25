#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""The module contains the helper functions for the hypermodels defined in hypermodels.py"""
# =============================================================================
# Imports
import tensorflow as tf
import numpy as np
import pandas as pd


#Choose a weight regularization (l1 and l2)
def get_regulizer_object(reg_name):
    """Returns a keras.regularizer object based on the provided reg_name parameter.

    Args:
        reg_name (string): Name of the regularizer we want to use: ["None", "l2_0.01", "l1_0.01", "l2_0.001", "l1_l2_0.01"]

    Returns:
        _type_: _description_
    """
    if reg_name == "None":
        return None
    elif reg_name == "l2_0.01":
        return tf.keras.regularizers.l2(0.01)
    elif reg_name == "l1_0.01":
        return tf.keras.regularizers.l1(0.01)
    elif reg_name == "l2_0.001":
        return tf.keras.regularizers.l2(0.001)
    elif reg_name == "l1_l2_0.01":
        return tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)

#Calculate class weights for the model from the training set
def calculate_class_weight(train_y):
    """Returns the class weight dictionary according to the distribution of each class in the training set.

    Args:
        train_y (pandas.series): The target values (y) for the binary classification problem.

    Returns:
        dict: Calculated weight for each class label:weight
    """
    pos = sum(train_y)
    neg = len(train_y) - pos
    total = pos + neg

    #Scaling the total by 2 will keep their magnitude the same
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight_ = {0: weight_for_0, 1: weight_for_1}
    
    return class_weight_

#Set the initial bias value of the final output layer according to the class distributions
def set_output_bias_initial_val(model, train_y):
    """Set the bias of the final sigmoid neuron according to the class distribution for a model.

    Args:
        model (keras.Model): An initialized model (that is not trained yet).
        train_y (pandas.series): The target values for the training set.

    Returns:
        keras.Model: The model with the final sigmoid neuron appropriately set.
    """
    pos = sum(train_y)
    neg = len(train_y) - pos
    initial_bias = np.log(pos/neg)
    
    config = model.get_config()
    config["layers"][-1]["config"]["bias_initializer"]["class_name"] = "Constant"
    config["layers"][-1]["config"]["bias_initializer"]["config"]['value'] = initial_bias
    model = model.from_config(config)
    return model

#Get the mean of a cross-validations folds for all the metrics
def get_mean_median_of_dicts(metric_dict_):
    """Calculates the mean and median of each metric for a given cross validation fold.
        Keras-tuner uses one of the values in this dic as the metric for the best hyperparameter.

    Args:
        metric_dict (dict): A dictionary of the metrics produced by keras.

    Returns:
        dict: Dict of mean and median of metrics for a given trial.
    """
    result_dict = dict()
    metrics_names = list(list(metric_dict_.values())[0].keys())
    
    for metric in metrics_names:
        metric_values_across_folds = [metric_dict_[trial][metric] for trial in metric_dict_]
        result_dict[f"mean_{metric}"] = np.mean(metric_values_across_folds)
        result_dict[f"median_{metric}"] = np.median(metric_values_across_folds)

    return result_dict
     
#Get an optimizer
def get_optimizer(optimizer_name):
    if optimizer_name == "Adam":
        return tf.keras.optimizers.Adam
    elif optimizer_name == "RMSProp":
        return tf.keras.optimizers.RMSprop
    else:
        raise "Wrond optimizer name."
        
#Get the last FUP array from all the FUPs
def get_last_FUPs_array(fups_X, timeseries_padding_value):
    """Creating an array of the the last FUPs.

    Args:
        fups_X (np.array): The fup array of shape sample:timestamps:features
        timeseries_padding_value (float): The float value used to pad the timeseries data.

    Returns:
        np.array: Last FUP arry of shape sample:features
    """
    new_array_list = []

    for sample in range(len(fups_X)):
        new_array_list.append([])
        for timeline in range(fups_X.shape[1]):
            if (fups_X[sample][timeline]!=np.array([timeseries_padding_value]*fups_X.shape[2], dtype='float32')).all():
                new_array_list[sample].append(fups_X[sample][timeline])
                
    final_FUP_array = []

    for patient_timelines in new_array_list:
        final_FUP_array.append(patient_timelines[-1]) #Extracting the final FUP
        
    return np.array(final_FUP_array)

#Record history metrics for the cross-validation        
def record_history_values(hypermodel, model, history_dict, metric_name, mode, fold_num, cv_results_dict, trial_metrics_dict, repeat_value):
    """Record the history metric of a given fold for a given trial (based on the metric_name and mode)
        in the cv_results_dict.

    Args:
        hypermodel (keras-tuner): The hypermodel object.
        model (keras.model): The trained model object.
        history_dict (dict): The history.history dict returned after the fit method in keras.
        metric_name (string): The name of the metric (ex: recall, precision, auc, etc.) that needs to be optimized.
        mode (string): Whether to choose the best hyperparameter as the max or min of metric.
        fold_num (int): The fold number for the given metric dic.
        cv_results_dict (dict): The cv results dict that stores all the metrics for all the folds and trials. 
        trial_metrics_dict (dict): The trial metric dic for a given trial.
    """
    
    #Find the index of the epoch at which the metric is optimized for a given fold cv
    if mode == "min":
        index = np.argmin(history_dict[metric_name])
    elif mode == "max":
        index = np.argmax(history_dict[metric_name])
    
    #Create a new dict which has all the metrics at the optimum epoch for that fold cv
    fold_metric = {key:values[index] for key, values in history_dict.items()}
    
    #Record the result of fold-cv in a dictionary for this trial
    trial_metrics_dict[f"repeat_{repeat_value}_fold_{fold_num}"] = fold_metric
    
    fold_metric_copy = fold_metric.copy()
                
    #Record the result of fold-cv in a dictionary provided to the hypermodel.fit for debugging/graphing
    fold_metric_copy["final_bias"] = model.get_config()["layers"][-1]["config"]["bias_initializer"]["config"]["values"]
    fold_metric_copy["best_epoch"] = index + 1
    cv_results_dict[f"trial_{hypermodel.trial_number}_repeat{repeat_value}_fold_{fold_num}"] = fold_metric_copy
    
    #Record the results in case there was a problem
    pd.DataFrame.from_dict(cv_results_dict, orient='index').to_csv(f"./keras_tuner_results/{hypermodel.name}/{hypermodel.name}_{hypermodel.unique_code}.csv")

