#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""The module contains the 4 keras-tuner hypermodel class for tuning."""
# =============================================================================
# Imports
import keras_tuner
import tensorflow as tf
from cross_validation import kfold_cv
from sklearn.dummy import DummyClassifier
from constants import timeseries_padding_value, NUMBER_OF_ITERATIONS_CV
from datetime import datetime
from hypermodel_helper import *
import copy
from joblib import dump, load
import time



class BaselineHyperModel(keras_tuner.HyperModel):
    """A fully connected tuner class for the baseline dataset.
    """
    
    def __init__(self, name):
        super().__init__()
        self.trial_number = 0 #Recording the number of trial
        self.cv_results_dict = dict() #Dict of cross-validation results.
        self.name = name #A name for this hypermodel
        self.unique_code = str(datetime.now().strftime("%D__%T").replace("/", "_").replace(":", "_"))
        
    def build(self, hp):
        
        #A sequential model
        model = tf.keras.Sequential()
        
        #Number of dense layers
        num_layers = hp.Int("num_dense_hidden_layers", 1, 3)
        
        #Number of nodes in the hidden layer(s)
        num_nodes = hp.Choice(f"hidden_layer_units", [1, 5, 10, 15, 20, 30, 40, 50, 100])
        
        #The choice of regularizer for the layer
        regularizer = hp.Choice(f'regularizer', ["None", "l1_0.01", "l2_0.01", "l1_l2_0.01"])
        
        #Dropout layer's rate
        dropout_rate = hp.Choice(f'dropout_rate', [0.10, 0.25, 0.50])
        
        #Creating the model
        for i in range(num_layers):
            with hp.conditional_scope('num_dense_hidden_layers', list(range(i + 1, 4))):
                model.add(
                    tf.keras.layers.Dense(
                        units=num_nodes,
                        name= f"Dense_{i}",
                        activation = 'relu',
                        kernel_regularizer = get_regulizer_object(regularizer)
                    )
                )
                #Adding dropout layer after each hidden layer
                model.add(tf.keras.layers.Dropout(rate=dropout_rate, name= f"Dropout_{i}"))
                  
        #Add the final output_layer to the model
        model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid"))
        
        return model
    
    def fit(self, hp, model, x, y, class_weight=None, grid_search=False, metric_name = None, metric_mode=None, feature_set_dict=None, callbacks=None, **kwargs):

        #Record the start time to calc total time
        start_time = time.time()
        
        #A dictionary that will store the validation metrics recorded for a given trial
        trial_metrics_dict = dict()
        
        #Get the training values from x
        baseline_train_val_X, fups_train_val_X = x
        
        #The choice of the feature set to use for the baseline dataset
        baseline_feature_choice = hp.Choice("baseline_feature_set", list(feature_set_dict["baseline_feature_sets"].keys()))
        features_index_to_keep_baseline = feature_set_dict["baseline_feature_sets"][baseline_feature_choice]
        baseline_train_val_X = baseline_train_val_X.iloc[:,features_index_to_keep_baseline]
                
        #The choice of optimizer
        optimizer = hp.Choice("optimizer", ["Adam", "RMSProp"])
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
            
        #Save the model config so that we can reset the model for each iterations/fold
        model_config = model.get_config()
        
        #Restricting the models which have the (num_nodes > 2*num_features) or (num_nodes < 2/3*num_features)
        num_nodes = hp.get("hidden_layer_units")
        num_features = len(features_index_to_keep_baseline)        
        if not (2/3*num_features <= num_nodes <= 2*num_features):
            
            print(f"Grid-Search Trial: {self.trial_number} omitted because num_nodes ({num_nodes}) and num_features ({num_features})")
            
            #Increment the trial number
            self.trial_number += 1 
            
            #return a zero dict
            return {f"mean_val_{metric_name}":-1, f"median_val_{metric_name}":-1}
            
        
        if grid_search:
            print(f"Performing Grid-Search Trial: {self.trial_number}")

            #Perform iterated k-fold cross-validation
            for repeat in range(1, NUMBER_OF_ITERATIONS_CV+1):
                #Perform 2-fold cross validation
                for fold_num, (training_data, validation_data) in enumerate(kfold_cv(baseline_train_val_X, fups_train_val_X, y, k=2, timeseries_padding_value=timeseries_padding_value, deterministic=False)):
                    baseline_train_X, _, train_y = training_data
                    baseline_valid_X, _, valid_y = validation_data
                                
                    #Reset the model by intializing it again. Otherwise, we will be training the same model over and over across each fold/iteration.
                    model = tf.keras.Sequential().from_config(model_config)
                    
                    #Change the bias weight of the final layer
                    model = set_output_bias_initial_val(model, train_y)

                    #Compile the model
                    model.compile(
                        optimizer=get_optimizer(optimizer)(learning_rate),
                        loss="binary_crossentropy",
                        metrics = [
                                    tf.keras.metrics.TruePositives(name='tp'),
                                    tf.keras.metrics.FalsePositives(name='fp'),
                                    tf.keras.metrics.TrueNegatives(name='tn'),
                                    tf.keras.metrics.FalseNegatives(name='fn'), 
                                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),
                                    tf.keras.metrics.AUC(name='auc'),
                                    tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
                                    ],
                    )

                    #Weight each class
                    class_weight = calculate_class_weight(train_y)
                                        
                    #Fit the model with the training and validate on validation sets
                    #The callback stops the training when there is no improvement on the validation set for 10 epochs
                    history = model.fit(
                        x = baseline_train_X,
                        y = train_y,
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor=f"val_{metric_name}", mode=metric_mode, patience=10)],
                        validation_data = (baseline_valid_X, valid_y),
                        class_weight = class_weight,
                        **kwargs
                        )
                    
                    
                    #Record the data obtained from this fold
                    record_history_values(
                                    hypermodel=self,
                                    model= model,
                                    history_dict=history.history,
                                    metric_name=f"val_{metric_name}",
                                    mode = metric_mode,
                                    fold_num = fold_num+1,
                                    cv_results_dict = self.cv_results_dict,
                                    trial_metrics_dict=trial_metrics_dict,
                                    repeat_value=repeat)
                    
                
            #Increment the trial number
            self.trial_number += 1 
            
            #Print total time spent
            print(f"Trial {self.trial_number-1} took {time.time()-start_time:.3f} seconds.")
            
            #Get the mean and median of the metrics, keras-tuner chooses either the mean or the median as instructed
            return get_mean_median_of_dicts(trial_metrics_dict)
        
        else:
            print(f"Performing final training. The data MUST be already normalized. Baseline feature set {baseline_feature_choice} will be used.")
                        
            #Change the bias weight of the final layer
            model = set_output_bias_initial_val(model, y)
            
            #Add weight to the class
            class_weight = calculate_class_weight(y)
            
            #Compile the model
            model.compile(
                optimizer = get_optimizer(optimizer)(learning_rate),
                loss="binary_crossentropy",
                metrics = [
                        tf.keras.metrics.TruePositives(name='tp'),
                        tf.keras.metrics.FalsePositives(name='fp'),
                        tf.keras.metrics.TrueNegatives(name='tn'),
                        tf.keras.metrics.FalseNegatives(name='fn'), 
                        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
                        ],
            )

            #Fit the model
            history = model.fit(
                x = baseline_train_val_X,
                y = y,
                class_weight = class_weight,
                **kwargs,
                        )
     
            return history, model

class LastFUPHyperModel(keras_tuner.HyperModel):
    
    def __init__(self, name):
        super().__init__()
        self.trial_number = 0 #Recording the number of trial
        self.cv_results_dict = dict() #Dict of cross-validation results.
        self.name = name #A name for this hypermodel
        self.unique_code = str(datetime.now().strftime("%D__%T").replace("/", "_").replace(":", "_"))
        
    def build(self, hp):
        
        #A sequential model
        model = tf.keras.Sequential()
        
        #Number of dense layers
        num_layers = hp.Int("num_dense_hidden_layers", 1, 3)
        
        #Number of nodes in the hidden layer(s)
        num_nodes = hp.Choice(f"hidden_layer_units", [12, 15, 20, 30, 40, 50, 90])
        
        #The choice of regularizer for the layer
        regularizer = hp.Choice(f'regularizer', ["None", "l1_0.01", "l2_0.01", "l1_l2_0.01"])
        
        #Dropout layer's rate
        dropout_rate = hp.Choice(f'dropout_rate', [0.10, 0.25, 0.50])
        
        #Creating the model
        for i in range(num_layers):
            with hp.conditional_scope('num_dense_hidden_layers', list(range(i + 1, 4))):
                model.add(
                    tf.keras.layers.Dense(
                        units=num_nodes,
                        name= f"Dense_{i}",
                        activation = 'relu',
                        kernel_regularizer = get_regulizer_object(regularizer)
                    )
                )
                #Adding dropout layer after each hidden layer
                model.add(tf.keras.layers.Dropout(rate=dropout_rate, name= f"Dropout_{i}"))
                  
        #Add the final output_layer to the model
        model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid"))
      
        return model
    
    def fit(self, hp, model, x, y, class_weight=None, grid_search=False, metric_name = None, metric_mode=None, feature_set_dict=None, callbacks=None, **kwargs):

        #Record the start time to calc total time
        start_time = time.time()
        
        #A dictionary that will store the validation metrics recorded for a given trial
        trial_metrics_dict = dict()
        
        #Get the training values from x
        baseline_train_val_X, fups_train_val_X = x
        
        #The choice of the feature set to use for FUP data
        fups_feature_choice = hp.Choice("FUPS_feature_set", list(feature_set_dict["FUPS_feature_sets"].keys()))
        features_index_to_keep_fups = feature_set_dict["FUPS_feature_sets"][fups_feature_choice]
        
        #The choice of optimizer and learning rate
        optimizer = hp.Choice("optimizer", ["Adam", "RMSProp"])
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        
        #Save the model config so that we can reset the model for each iterations/fold
        model_config = model.get_config()
        
        #Restricting the models which have the (num_nodes > 2*num_features) or (num_nodes < 2/3*num_features)
        num_nodes = hp.get("hidden_layer_units")
        num_features = len(features_index_to_keep_fups)        
        if not (2/3*num_features <= num_nodes <= 2*num_features):
            
            print(f"Grid-Search Trial: {self.trial_number} omitted because when num_nodes ({num_nodes}) and num_features ({num_features})")
            
            #Increment the trial number
            self.trial_number += 1 
            
            #return a zero dict
            return {f"mean_val_{metric_name}":-1, f"median_val_{metric_name}":-1}

        if grid_search:
            print(f"Performing Grid-Search Trial: {self.trial_number}")

            #Perform repeated k-fold cross-validation
            for repeat in range(1, NUMBER_OF_ITERATIONS_CV+1):
                #Perform 2-fold cross validation
                for fold_num, (training_data, validation_data) in enumerate(kfold_cv(baseline_train_val_X, fups_train_val_X, y, k=2, timeseries_padding_value=timeseries_padding_value, deterministic=False)):
                    _, fups_train_X, train_y = training_data
                    _, fups_valid_X, valid_y = validation_data
                            
                    #Only use the appropriate feature set
                    fups_train_X = fups_train_X[:,:,features_index_to_keep_fups]
                    fups_valid_X = fups_valid_X[:,:,features_index_to_keep_fups]
                    
                    
                    #Only keep the final FUP vector
                    fups_train_X = get_last_FUPs_array(fups_X=fups_train_X, timeseries_padding_value=timeseries_padding_value)
                    fups_valid_X = get_last_FUPs_array(fups_X=fups_valid_X, timeseries_padding_value=timeseries_padding_value)
                    
                    #Reset the model by intializing it again. Otherwise, we will be training the same model over and over across each fold/iteration.
                    model = tf.keras.Sequential().from_config(model_config)
                    
                    #Change the bias weight of the final layer
                    model = set_output_bias_initial_val(model, train_y)

                    #Compile the model
                    model.compile(
                        optimizer=get_optimizer(optimizer)(learning_rate=learning_rate),
                        loss="binary_crossentropy",
                        metrics = [
                                    tf.keras.metrics.TruePositives(name='tp'),
                                    tf.keras.metrics.FalsePositives(name='fp'),
                                    tf.keras.metrics.TrueNegatives(name='tn'),
                                    tf.keras.metrics.FalseNegatives(name='fn'), 
                                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),
                                    tf.keras.metrics.AUC(name='auc'),
                                    tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
                                    ],
                    )

                    #Weight each class
                    class_weight = calculate_class_weight(train_y)
                    
                    #Fit the model with the training and validate on validation sets
                    #The callback stops the training when there is no improvement on the validation set for 8 epochs
                    history = model.fit(
                        x = fups_train_X,
                        y = train_y,
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor=f"val_{metric_name}", mode=metric_mode, patience=10)],
                        validation_data = (fups_valid_X, valid_y),
                        class_weight = class_weight,
                        **kwargs
                        )
                    
                    
                    #Record the data obtained from this fold
                    record_history_values(
                                    hypermodel=self,
                                    model= model,
                                    history_dict=history.history,
                                    metric_name=f"val_{metric_name}",
                                    mode = metric_mode,
                                    fold_num = fold_num+1,
                                    cv_results_dict = self.cv_results_dict,
                                    trial_metrics_dict=trial_metrics_dict,
                                    repeat_value=repeat)
                    
                
            #Increment the trial number
            self.trial_number += 1 
            
            #Print total time spent
            print(f"Trial {self.trial_number-1} took {time.time()-start_time:.3f} seconds.")
            
            #Get the mean and median of the metrics, keras-tuner chooses either the mean of the median as instructed
            return get_mean_median_of_dicts(trial_metrics_dict)
        
        else:
            print(f"Performing final training. The data MUST be already normalized. Baseline feature set {fups_feature_choice} will be used.")
                          
            #Keeping the appropriate feature choice                
            fups_train_val_X = fups_train_val_X[:,:,features_index_to_keep_fups]

            #Only keep the final FUP vector
            fups_train_val_X = get_last_FUPs_array(fups_X=fups_train_val_X, timeseries_padding_value=timeseries_padding_value)
        
            #Change the bias weight of the final layer
            model = set_output_bias_initial_val(model, y)
            
            #Add weight to the class
            class_weight = calculate_class_weight(y)
            
            #Compile the model
            model.compile(
                optimizer=get_optimizer(optimizer)(learning_rate),
                loss="binary_crossentropy",
                metrics = [
                        tf.keras.metrics.TruePositives(name='tp'),
                        tf.keras.metrics.FalsePositives(name='fp'),
                        tf.keras.metrics.TrueNegatives(name='tn'),
                        tf.keras.metrics.FalseNegatives(name='fn'), 
                        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
                        ],
            )

            #Fit the model
            history = model.fit(
                x = fups_train_val_X,
                y = y,
                class_weight = class_weight,
                **kwargs,
                        )
     
            return history, model

class FUP_RNN_HyperModel(keras_tuner.HyperModel):
    
    def __init__(self, name):
        super().__init__()
        self.trial_number = 0 #Recording the number of trial
        self.cv_results_dict = dict() #Dict of cross-validation results.
        self.name = name #A name for this hypermodel
        self.unique_code = str(datetime.now().strftime("%D__%T").replace("/", "_").replace(":", "_"))
        
    def build(self, hp):
        
        #A sequential model
        model = tf.keras.Sequential()
        
        #Add the Masking layer
        model.add(tf.keras.layers.Masking(mask_value=timeseries_padding_value))
        
        #Number of RNN layers
        num_layers = hp.Int("num_LSTM_layers", 1, 2)
        
        #Number of nodes in the hidden layer(s)
        num_nodes = hp.Choice(f"LSTM_layer_units", [12, 15, 20, 30, 40, 50, 90])
        
        #The choice of regularizer for the layer
        regularizer = hp.Choice(f'regularizer', ["None", "l1_0.01", "l2_0.01", "l1_l2_0.01"])
        
        #Dropout layer's rate
        recurrent_dropout_rate = hp.Choice(f'recurrent_dropout_rate', [0.10, 0.25, 0.50])
        
        for i in range(num_layers):
            with hp.conditional_scope('num_LSTM_layers', list(range(i+1, 3))):
                model.add(
                    tf.keras.layers.LSTM(
                        units=num_nodes,
                        name= f"LSTM_{i}",
                        recurrent_dropout = recurrent_dropout_rate,
                        activation = 'tanh',
                        return_sequences = (num_layers > 1) & (i != num_layers-1), #If more than one layer, return sequence
                        kernel_regularizer = get_regulizer_object(regularizer)
                    )
                )
                
        #Adding the final dropout layer
        model.add(tf.keras.layers.Dropout(rate=hp.Choice('final_dropout_rate', [0.1, 0.25, 0.50]), name="final_dropout_layer"))
            
        #Add the final output_layer to the model
        model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid"))
        
        return model
    
    def fit(self, hp, model, x, y, class_weight=None, grid_search=False, metric_name = None, metric_mode=None, feature_set_dict=None, callbacks=None, **kwargs):

        #Record the start time to calc total time
        start_time = time.time()
                            
        #A dictionary that will store the validation metrics recorded for a given trial
        trial_metrics_dict = dict()
        
        #Get the training values from x
        baseline_train_val_X, fups_train_val_X = copy.deepcopy(x[0]),  copy.deepcopy(x[1])
        
        #The choice of the feature set to use for FUP data
        fups_feature_choice = hp.Choice("FUPS_feature_set", list(feature_set_dict["FUPS_feature_sets"].keys()))
        features_index_to_keep_fups = feature_set_dict["FUPS_feature_sets"][fups_feature_choice]
        
        #The choice of optimizer and learning rate
        optimizer = hp.Choice("optimizer", ["Adam", "RMSProp"])
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        
        #Save the model config so that we can reset the model for each iterations/fold
        model_config = model.get_config()
        
        #Restricting the models which have the (num_nodes > 2*num_features) or (num_nodes < 2/3*num_features)
        num_nodes = hp.get("LSTM_layer_units")
        num_features = len(features_index_to_keep_fups)    
        if not (2/3*num_features <= num_nodes <= 2*num_features):
            
            print(f"Grid-Search Trial: {self.trial_number} omitted because num_nodes ({num_nodes}) and num_features ({num_features})")
            
            #Increment the trial number
            self.trial_number += 1 
            
            #return a zero dict
            return {f"mean_val_{metric_name}":-1, f"median_val_{metric_name}":-1}

        if grid_search:
            print(f"Performing Grid-Search Trial: {self.trial_number}")

            #Perform repeated k-fold cross-validation
            for repeat in range(1, NUMBER_OF_ITERATIONS_CV+1):
                #Perform 2-fold cross validation
                for fold_num, (training_data, validation_data) in enumerate(kfold_cv(baseline_train_val_X, fups_train_val_X, y, k=2, timeseries_padding_value=timeseries_padding_value, deterministic=False)):
                    _, fups_train_X, train_y = training_data
                    _, fups_valid_X, valid_y = validation_data   
                    
                    #Keep only the features that were selected
                    fups_train_X = fups_train_X[:,:,features_index_to_keep_fups]   
                    fups_valid_X = fups_valid_X[:,:,features_index_to_keep_fups]   
                                
                    #Reset the model by intializing it again. Otherwise, we will be training the same model over and over across each fold/iteration.
                    model = tf.keras.Sequential().from_config(model_config)
                    
                    #Change the bias weight of the final layer
                    model = set_output_bias_initial_val(model, train_y)
                    
                    #Compile the model
                    model.compile(
                        optimizer = get_optimizer(optimizer)(learning_rate),
                        loss = "binary_crossentropy",
                        metrics = [
                                    tf.keras.metrics.TruePositives(name='tp'),
                                    tf.keras.metrics.FalsePositives(name='fp'),
                                    tf.keras.metrics.TrueNegatives(name='tn'),
                                    tf.keras.metrics.FalseNegatives(name='fn'), 
                                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),
                                    tf.keras.metrics.AUC(name='auc'),
                                    tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
                                    ],
                    )

                    #Weight each class
                    class_weight = calculate_class_weight(train_y)
                    
                    #Fit the model with the training and validate on validation sets
                    #The callback stops the training when there is no improvement on the validation set for 8 epochs
                    history = model.fit(
                        x = fups_train_X,
                        y = train_y,
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor=f"val_{metric_name}", mode=metric_mode, patience=10)],
                        validation_data = (fups_valid_X, valid_y),
                        class_weight = class_weight,
                        **kwargs
                        )
                    
                    
                    #Record the data obtained from this fold
                    record_history_values(
                                    hypermodel=self,
                                    model= model,
                                    history_dict=history.history,
                                    metric_name=f"val_{metric_name}",
                                    mode = metric_mode,
                                    fold_num = fold_num+1,
                                    cv_results_dict = self.cv_results_dict,
                                    trial_metrics_dict=trial_metrics_dict,
                                    repeat_value=repeat)
                    
                
            #Increment the trial number
            self.trial_number += 1 
            
            #Print total time spent
            print(f"Trial {self.trial_number-1} took {time.time()-start_time:.3f} seconds.")
            
            #Get the mean and median of the metrics, keras-tuner chooses either the mean of the median as instructed
            return get_mean_median_of_dicts(trial_metrics_dict)
        
        else:
            print(f"Performing final training. The data MUST be already normalized. Baseline feature set {fups_feature_choice} will be used.")

            #Keep only the features that were selected
            fups_train_val_X = fups_train_val_X[:,:,features_index_to_keep_fups]  
            
            #Change the bias weight of the final layer
            model = set_output_bias_initial_val(model, y)
            
            #Add weight to the class
            class_weight = calculate_class_weight(y)
            
            #Compile the model
            model.compile(
                optimizer=get_optimizer(optimizer)(learning_rate),
                loss="binary_crossentropy",
                metrics = [
                        tf.keras.metrics.TruePositives(name='tp'),
                        tf.keras.metrics.FalsePositives(name='fp'),
                        tf.keras.metrics.TrueNegatives(name='tn'),
                        tf.keras.metrics.FalseNegatives(name='fn'), 
                        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
                        ],
            )

            #Fit the model
            history = model.fit(
                x = fups_train_val_X,
                y = y,
                class_weight = class_weight,
                **kwargs,
                        )
                 
            return history, model

class Baseline_FUP_Multiinput_HyperModel(keras_tuner.HyperModel):
    
    def __init__(self, name):
        super().__init__()
        self.trial_number = 0 #Recording the number of trial
        self.cv_results_dict = dict() #Dict of cross-validation results.
        self.name = name #A name for this hypermodel
        self.unique_code = str(datetime.now().strftime("%D__%T").replace("/", "_").replace(":", "_"))
        
    def build(self, hp):
        
        #############################
        #Baseline dataset fully connected network
        
        #Baseline input shape (Just initializing with 1000 dimentions)
        input_baseline = tf.keras.layers.Input(shape=(1000,), dtype="float32", name="baseline_input")

        #Number of dense hidden layers in baseline
        num_dense_layers_baseline = hp.Int("num_dense_hidden_layers_baseline", 1, 3)
        
        #Number of nodes in the hidden layer(s)
        num_nodes_hidden_baseline = hp.Choice(f"hidden_layer_units_baseline", [1, 5, 10, 15, 20, 30, 40, 50, 100])
        
        #The choice of regularizer for the layer
        regularizer_hidden_baseline = hp.Choice(f'regularizer_hidden_baseline', ["None", "l1_0.01", "l2_0.01", "l1_l2_0.01"])
        
        #Dropout layer's rate
        dropout_rate_hidden_baseline = hp.Choice(f'dropout_rate_hidden_baseline', [0.10, 0.25, 0.50])
        
        #Adding 2 to 5 hidden layers where each layer has 5 - 40 nodes (this is for grid search with keras-tuner)
        for i in range(num_dense_layers_baseline):
            with hp.conditional_scope('num_dense_hidden_layers_baseline', list(range(i + 1, 4))):
                #Define a layer
                dense = tf.keras.layers.Dense(
                    units=num_nodes_hidden_baseline,
                    name= f"Baseline_Dense_{i}",
                    activation = 'relu',
                    kernel_regularizer = get_regulizer_object(regularizer_hidden_baseline)
                )
                
                if i == 0:
                    x = dense(input_baseline)
                else:
                    x = dense(x)
                
                #Adding dropout layer after each hidden layer
                x = tf.keras.layers.Dropout(rate=dropout_rate_hidden_baseline, name= f"Dropout_Dense_layer_baseline_{i}")(x)
        
        ####################################
        #Fup dataset RNN  
        
        #FUP input shape (Just initializing with 1000 dimentions)
        fup_input = tf.keras.layers.Input(shape=(None,1000), dtype="float32", name="fup_input")
        
        #Add a masking layer
        mask = tf.keras.layers.Masking(mask_value=timeseries_padding_value)(fup_input)
        
        #Number of RNN layer
        num_layers_LSTM_fup = hp.Int("num_layers_LSTM_fup", 1, 2)
        
        #Number of nodes in the hidden layer(s)
        num_nodes_LSTM_FUP = hp.Choice("LSTM_layer_units", [12, 15, 20, 30, 40, 50, 90])
        
        #The choice of regularizer for the layer
        regularizer_LSTM_FUP = hp.Choice('regularizer_LSTM_FUP', ["None", "l1_0.01", "l2_0.01", "l1_l2_0.01"])
        
        #Dropout layer's rate
        recurrent_dropout_rate = hp.Choice('recurrent_dropout_rate_LSTM', [0.10, 0.25, 0.50])
        
        #Adding 1 or 2 hidden layers where each layer has 5 - 50 nodes
        for i in range(num_layers_LSTM_fup):
            with hp.conditional_scope('num_layers_LSTM_fup', list(range(i+1, 3))):
                lstm = tf.keras.layers.LSTM(
                        units=num_nodes_LSTM_FUP,
                        name= f"LSTM_{i}",
                        recurrent_dropout = recurrent_dropout_rate,
                        activation = 'tanh',
                        return_sequences = (num_layers_LSTM_fup > 1) & (i != num_layers_LSTM_fup-1), #If more than one layer, return sequence
                        kernel_regularizer = get_regulizer_object(regularizer_LSTM_FUP)
                    )
                
                if i == 0:
                    y = lstm(mask)
                else:
                    y = lstm(y)
    
        ######################################
        #Building the final multiinput model
        
        #Concatenate the output of the dense (baseline) and LSTM (FUP) 
        xy = tf.keras.layers.concatenate([x, y])
        
        #Number of nodes in the final dense layer
        num_final_nodes = (num_nodes_hidden_baseline+num_nodes_LSTM_FUP)//2
        
        #Final dropout rate
        final_dropout_rate = hp.Choice('final_dropout_rate', [0.1, 0.25])
        
        
        dense = tf.keras.layers.Dense(
            units=num_final_nodes,
            name= f"Final_Dense_{i}",
            activation = 'relu',
        )
                
        
        x = dense(xy)
        x = tf.keras.layers.Dropout(rate=final_dropout_rate, name= "dropout_after_final_dense_layer")(x)

                
        final_output = tf.keras.layers.Dense(1, activation="sigmoid", name="final_output")(x)
        
        model = tf.keras.Model(inputs=[input_baseline, fup_input], outputs=final_output)
        
        return model
    
    def fit(self, hp, model, x, y, class_weight=None, grid_search=False, metric_name = None, metric_mode=None, feature_set_dict=None, callbacks=None, **kwargs):
        
        #Record the start time to calc total time
        start_time = time.time()
        
        def set_input_shape_of_model(model, baseline_train_X, fups_train_X):
            #Change the input shape according to the feature sets
            model_config = model.get_config().copy()

            for i, layer in enumerate(model_config["layers"]):
                if layer["name"] == "baseline_input":
                    model_config["layers"][i]["config"]["batch_input_shape"] = (None, baseline_train_X.shape[-1])

            for i, layer in enumerate(model_config["layers"]):
                if layer["name"] == "fup_input":
                    model_config["layers"][i]["config"]["batch_input_shape"] = (None, None, fups_train_X.shape[-1])
                    
            return model.from_config(model_config)
        
        #A dictionary that will store the validation metrics recorded for a given trial
        trial_metrics_dict = dict()
        
        #Get the training values from x
        baseline_train_val_X, fups_train_val_X =  copy.deepcopy(x[0]),  copy.deepcopy(x[1])
        
        #The choice of the feature set to use for the baseline dataset
        baseline_feature_choice = hp.Choice("baseline_feature_set", list(feature_set_dict["baseline_feature_sets"].keys()))
        features_index_to_keep_baseline = feature_set_dict["baseline_feature_sets"][baseline_feature_choice]
        baseline_train_val_X = baseline_train_val_X.iloc[:,features_index_to_keep_baseline]
        
        #The choice of the feature set to use for FUP data
        fups_feature_choice = hp.Choice("FUPS_feature_set", list(feature_set_dict["FUPS_feature_sets"].keys()))
        features_index_to_keep_fups = feature_set_dict["FUPS_feature_sets"][fups_feature_choice]
        
                
        #The choice of optimizer and learning rate
        optimizer = hp.Choice("optimizer", ["Adam", "RMSProp"])
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        
        #Save the model config so that we can reset the model for each iterations/fold
        model_config = model.get_config()
        
        #Restricting the models which have the (num_nodes > 2*num_features) or (num_nodes < 2/3*num_features)
        num_nodes_baseline = hp.get("hidden_layer_units_baseline")
        num_nodes_fups = hp.get("LSTM_layer_units")
        
        num_features_baseline = len(features_index_to_keep_baseline)
        num_features_fups = len(features_index_to_keep_fups)
        
              
        if not (2/3*num_features_fups <= num_nodes_fups <= 2*num_features_fups) or not (2/3*num_features_baseline <= num_nodes_baseline <= 2*num_features_baseline):
            
            print(f"Grid-Search Trial: {self.trial_number} omitted because \
                  num_nodes_baseline: {num_nodes_baseline} |  num_features_baseline {num_features_baseline} \
                  num_nodes_fups: {num_nodes_fups} | num_features_fups: {num_features_fups}")
            
            #Increment the trial number
            self.trial_number += 1 
            
            #return a zero dict
            return {f"mean_val_{metric_name}":-1, f"median_val_{metric_name}":-1}
        
        
        if grid_search:
            print(f"Performing Grid-Search Trial: {self.trial_number}")

            #Perform repeated k-fold cross-validation
            for repeat in range(1, NUMBER_OF_ITERATIONS_CV+1):
                #Perform 2-fold cross validation
                for fold_num, (training_data, validation_data) in enumerate(kfold_cv(baseline_train_val_X, fups_train_val_X, y, k=2, timeseries_padding_value=timeseries_padding_value, deterministic=False)):
                    baseline_train_X, fups_train_X, train_y = training_data
                    baseline_valid_X, fups_valid_X, valid_y = validation_data
                    
                    #Only keep the selected FUP feature (we don't need to do that for baseline as it is done above)
                    fups_train_X = fups_train_X[:,:,features_index_to_keep_fups]
                    fups_valid_X = fups_valid_X[:,:,features_index_to_keep_fups]
                    
                    #Reset the model by intializing it again. Otherwise, we will be training the same model over and over across each fold/iteration.
                    model = tf.keras.Model().from_config(model_config)
                                 
                    #Change the bias weight of the final layer
                    model = set_output_bias_initial_val(model, train_y)
                    
                    #Change the shape of the input layers for both the baseline dense and lstm
                    model = set_input_shape_of_model(model, baseline_train_X, fups_train_X)
                    
                    #Compile the model
                    model.compile(
                        optimizer = get_optimizer(optimizer)(learning_rate),
                        loss = "binary_crossentropy",
                        metrics = [
                                    tf.keras.metrics.TruePositives(name='tp'),
                                    tf.keras.metrics.FalsePositives(name='fp'),
                                    tf.keras.metrics.TrueNegatives(name='tn'),
                                    tf.keras.metrics.FalseNegatives(name='fn'), 
                                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),
                                    tf.keras.metrics.AUC(name='auc'),
                                    tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
                                    ],
                    )

                    #Weight each class
                    class_weight = calculate_class_weight(train_y)
                    
                    #Fit the model with the training and validate on validation sets
                    #The callback stops the training when there is no improvement on the validation set for 8 epochs
                    history = model.fit(
                        x = [baseline_train_X, fups_train_X],
                        y = train_y,
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor=f"val_{metric_name}", mode=metric_mode, patience=10)],
                        validation_data = ([baseline_valid_X ,fups_valid_X], valid_y),
                        class_weight = class_weight,
                        **kwargs
                        )
                    
                    
                    #Record the data obtained from this fold
                    record_history_values(
                                    hypermodel=self,
                                    model= model,
                                    history_dict=history.history,
                                    metric_name=f"val_{metric_name}",
                                    mode = metric_mode,
                                    fold_num = fold_num+1,
                                    cv_results_dict = self.cv_results_dict,
                                    trial_metrics_dict=trial_metrics_dict,
                                    repeat_value=repeat)
                    
                
            #Increment the trial number
            self.trial_number += 1 
            
            #Print total time spent
            print(f"Trial {self.trial_number-1} took {time.time()-start_time:.3f} seconds.")
            
            #Get the mean and median of the metrics, keras-tuner chooses either the mean of the median as instructed
            return get_mean_median_of_dicts(trial_metrics_dict)
        
        else:
            print(f"Performing final training. The data MUST be already normalized. Baseline feature \
                  set {baseline_feature_choice} and FUPS feature set {fups_feature_choice} will be used.")     
            
            fups_train_val_X = fups_train_val_X[:,:,features_index_to_keep_fups]
           
        
            #Change the bias weight of the final layer
            model = set_output_bias_initial_val(model, y)
            
            #Change the shape of the input layers for both the baseline dense and lstm
            model = set_input_shape_of_model(model, baseline_train_val_X, fups_train_val_X)
            
            #Add weight to the class
            class_weight = calculate_class_weight(y)
            
            #Compile the model
            model.compile(
                optimizer=get_optimizer(optimizer)(learning_rate),
                loss="binary_crossentropy",
                metrics = [
                        tf.keras.metrics.TruePositives(name='tp'),
                        tf.keras.metrics.FalsePositives(name='fp'),
                        tf.keras.metrics.TrueNegatives(name='tn'),
                        tf.keras.metrics.FalseNegatives(name='fn'), 
                        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
                        ],
            )

            #Fit the model
            history = model.fit(
                x = [baseline_train_val_X, fups_train_val_X],
                y = y,
                class_weight = class_weight,
                **kwargs,
                        )
     
            return history, model


def dummy_classifiers(X_train, y_train, X_test, y_test, FUPS_dict):
    """Trains and tests 3 types of dummy classifiers and saves their performance metrics.

    Args:
        X_train (numpy.array): X_train
        y_train (numpy.array): y_train
        X_test (numpy.array): X_test
        y_test (numpy.array): y_test
        FUPS_dict (dict): The dictionary of the FUP data. Keys are the ids, and values are 2D array of (timeline, features).

    """
    
    all_data_dics = dict()

    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    
    dummy_clf_most_frequent = DummyClassifier(strategy="stratified", random_state=1375)
    dummy_clf_most_frequent.name = "stratified"
    
    dummy_clf_prior = DummyClassifier(strategy="prior", random_state=1375)
    dummy_clf_prior.name = "most_frequent"
    
    dummy_clf_uniform = DummyClassifier(strategy="uniform", random_state=1375)
    dummy_clf_uniform.name = "uniform"
    
    for clf_object in [dummy_clf_most_frequent, dummy_clf_prior, dummy_clf_uniform]:
    
        #train the clf
        clf_object.fit(X_train, y_train)
        
        #Save the model
        dump(clf_object, f"./keras_tuner_results/Dummy_classifiers/{clf_object.name}_model.joblib")
        #Load the model
        
        clf_object = load(f"./keras_tuner_results/Dummy_classifiers/{clf_object.name}_model.joblib")
        
        #for each clf, test on the training and testing set then save the metrics.
        for x, y, name in zip([X_train, X_test], [y_train, y_test], ["training", "testing"]):
            
            #y_pred
            y_pred = clf_object.predict(x)
            
            #Record exactly what are the predictions for each sample on the test dataset
            if name == "testing":
                y_pred_classes = (y_pred > 0.5).astype("int32").flatten()
                number_of_FUP = [len(FUPS_dict[uniqid]) for uniqid in list(y.index)]
                record_dict = {"uniqid":list(y.index),"FUP_numbers":number_of_FUP, "y_actual":y.values,
                            "y_pred":y_pred.flatten(), "y_pred_classes":y_pred_classes}

                pd.DataFrame(record_dict).to_csv(f"keras_tuner_results/Dummy_classifiers/{clf_object.name}_detailed_test_results.csv")

                    
            #Metric
            metric_dict = dict()
            
            for metric in METRICS:
                metric_value = metric(y, y_pred).numpy()
                metric_name = metric.name
                metric_dict[metric_name] = metric_value
                metric.reset_state()
                
            m = tf.keras.metrics.AUC()
            m.update_state(y, y_pred)

            # Set `x` and `y` values for the curves based on `curve` config.
            recall = tf.math.divide_no_nan(
                m.true_positives,
                tf.math.add(m.true_positives, m.false_negatives))

            fp_rate = tf.math.divide_no_nan(
                    m.false_positives,
                    tf.math.add(m.false_positives, m.true_negatives))

            precision = tf.math.divide_no_nan(
                    m.true_positives,
                    tf.math.add(m.true_positives, m.false_positives))
            
            metric_dict["precision_curve"] = precision.numpy()
            metric_dict["fp_rate"] = fp_rate.numpy()
            metric_dict["recall_curve"] = recall.numpy()
            metric_dict["thresholds"] = m.thresholds
            
            all_data_dics[name+"_"+clf_object.name] = metric_dict

    pd.DataFrame.from_dict(all_data_dics, orient="index").to_pickle(f"./keras_tuner_results/Dummy_classifiers/Dummy_clfs_train_test_results.pkl")
        