from tensorflow import keras
import keras_tuner
import tensorflow as tf
from imblearn.over_sampling import SMOTE, RandomOverSampler 
import numpy as np
import pandas as pd
from cross_validation import kfold_cv
import matplotlib.pyplot as plt

class HyperModelBaseline(keras_tuner.HyperModel):
    
    #We want to save the number of trials that has occured, 
    #also record the metrics for each fold of each trial (debugging)
    def __init__(self, model_type, timeseries_padding_value):
        super().__init__()
        self.trial_number = 0 #
        self.cv_results_dict = dict() #Dic of cross-validation results.
        
        #List of metrics we record
        self.METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'), 
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
            ]
        
        self.model_type = model_type #Name of the architecture could be "Dense, RNN, Dense_RNN"
        self.timeseries_padding_value = timeseries_padding_value #Padding value used in RNN for timeseries data.
    
    def build(self, hp):
        
        #The weights are initialized with a defined seed for reproducibility
        tf.random.set_seed(1375)
        
        def get_RNN_layer(layer_type, i=1):
            """Returns a GRU or LSTM layer with KerasTuner HyperModel hyperparameter annotations.

            Args:
                layer_type (string): Could be either "GRU" or "LSTM".
                i (int, optional): The layer number in the sequence. Defaults to 1.

            Returns:
                Keras.layers: A GRU or LSTM object.
            """
            if layer_type == "GRU":
                layer = keras.layers.GRU(units=hp.Int(f"units_{i}", min_value=5, max_value=80, step=5), 
                                activation="tanh", name= "RNN_layer",
                                recurrent_dropout=hp.Choice(f'dropout_rate_layer{i}', [0.0, 0.25, 0.50]))
            elif layer_type == "LSTM":
                layer = keras.layers.LSTM(units=hp.Int(f"units_{i}", min_value=5, max_value=80, step=5), 
                                activation="tanh",name= "RNN_layer",
                                recurrent_dropout=hp.Choice(f'dropout_rate_layer{i}', [0.0, 0.25, 0.50]))
            
            return layer
          
        #Choose a weight regularization (l1 and l2)
        def get_regulizer_object(reg_name):
            """Returns a keras.regularizer object based on the provided reg_name parameter.

            Args:
                reg_name (string): Name of the regularizer we want to use: ["None", "l2_0.01", "l1_0.01", "l2_0.002", "l1_l2_0.001"]

            Returns:
                _type_: _description_
            """
            if reg_name == "None":
                return None
            elif reg_name == "l2_0.01":
                return keras.regularizers.l2(0.01)
            elif reg_name == "l1_0.01":
                return keras.regularizers.l1(0.01)
            elif reg_name == "l2_0.002":
                return keras.regularizers.l2(0.02)
            elif reg_name == "l1_l2_0.001":
                return keras.regularizers.l1_l2(l1=0.001, l2=0.001)
        
        if self.model_type == "Dense":

            #A sequential model
            model = keras.Sequential()
            
            #Adding 1 to 3 hidden layers where each layer has 5 - 90 nodes (this is for grid search with keras-tuner)
            for i in range(hp.Int("num_layers", 1, 3)):
                model.add(
                    keras.layers.Dense(
                        units=hp.Int(f"units_{i}", min_value=5, max_value=90, step=5),
                        activation = 'relu',
                        kernel_regularizer = get_regulizer_object(hp.Choice(f'regulaizer_layer{i}', ["None", "l2_0.01", "l2_0.002", "l1_l2_0.001", "l1_0.01"]))
                    )
                )
                #Adding dropout layer after each hidden layer with 0.25 or 0.50 rate.
                if hp.Boolean(f'dropout_after_layer{i}'):
                    with hp.conditional_scope(f'dropout_after_layer{i}', [True]):
                        model.add(keras.layers.Dropout(rate=hp.Choice(f'dropout_rate_layer{i}', [0.25, 0.50])))
                    
            #Add the final output_layer to the model
            model.add(keras.layers.Dense(1, activation="sigmoid", name="output_layer"))
            
        elif self.model_type == "RNN":
            
            #A sequential model
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(None, 45)))
            model.add(keras.layers.Masking(mask_value=self.timeseries_padding_value))
            
            RNN_layer_type = hp.Choice("layer_type", ["GRU", "LSTM"])

            #Adding 1 RNN layer
            model.add(
                get_RNN_layer(RNN_layer_type)
            )
                
            #Adding dropout layer before the final layer
            if hp.Boolean('dropout_after_final_layer'):
                with hp.conditional_scope('dropout_after_final_layer', [True]):
                    model.add(keras.layers.Dropout(rate=hp.Choice(f'dropout_rate_final_layer', [0.25, 0.50])))
                    
            #Add the final output_layer to the model
            model.add(keras.layers.Dense(1, activation="sigmoid", name="output_layer"))

        elif self.model_type == "Dense_RNN":
            #The model with a dense layer for BASELINE data, and the RNN layer for the Follow-up data
            
            #The input of the baseline dataset
            baseline_input = keras.Input(shape=(100,), name="baseline_input")


            #Adding 1 to 3 hidden layers where each layer has 5 - 90 nodes (this is for grid search with keras-tuner)
            for i in range(hp.Int("num_layers", 1, 3)):
                baseline_dense_layer = keras.layers.Dense(
                        units=hp.Int(f"units_{i}", min_value=5, max_value=90, step=5),
                        activation = 'relu',
                        kernel_regularizer = get_regulizer_object(hp.Choice(f'regulaizer_layer{i}', ["None", "l2_0.01", "l2_0.002", "l1_l2_0.001", "l1_0.01"]))
                            )
                
                #If it is the first layer, we add the input layer, otherwise we add the baseline dense
                if i == 0: 
                    baseline_dense = baseline_dense_layer(baseline_input)
                else:
                    baseline_dense = baseline_dense_layer(baseline_dense)
                
                #Adding dropout layer after each hidden layer with 0.25 or 0.50 rate.
                if hp.Boolean(f'dropout_after_baseline_layer{i}'):
                    with hp.conditional_scope(f'dropout_after_baseline_layer{i}', [True]):
                        baseline_dense = keras.layers.Dropout(rate=hp.Choice(f'dropout_rate_baseline_layer{i}', [0.25, 0.50]))(baseline_dense)
              

            fup_input = keras.Input(shape=(None, 45), name="RNN_input")
            fup_mask = keras.layers.Masking(mask_value=self.timeseries_padding_value, name="RNN_mask")(fup_input)

            RNN_layer_type = hp.Choice("layer_type", ["GRU", "LSTM"])

            #Adding 1
            fup_lstm = get_RNN_layer(RNN_layer_type)
            fup_lstm = fup_lstm(fup_mask)
                
            #Adding dropout layer before the final layer
            if hp.Boolean('dropout_after_final_RNN_layer'):
                with hp.conditional_scope('dropout_after_final_RNN_layer', [True]):
                    fup_lstm = keras.layers.Dropout(rate=hp.Choice(f'dropout_rate_final_RNN_layer', [0.25, 0.50]))(fup_lstm)
                    

            x = keras.layers.Concatenate()([baseline_dense, fup_lstm])

            final_layer = keras.layers.Dense(1, activation="sigmoid", name="output_layer")(x)

            model = keras.Model(inputs=[baseline_input, fup_input], outputs=final_layer)
        
        return model
    
    def fit(self, hp, model, x, y, class_weight=None, callbacks=None, grid_search=False, metric_name = None, metric_mode=None, baseline_feature_dic=None,  **kwargs):
        
        #Calculate class weights for the model from the training set
        def calculate_class_weight(train_y):
            pos = sum(train_y)
            neg = len(train_y) - pos
            total = pos + neg

            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)

            class_weight_ = {0: weight_for_0, 1: weight_for_1}
            
            return class_weight_
        
        #Set the initial bias value of the final output layer according to the class distributions
        def set_output_bias_initial_val(model, train_y):
            pos = sum(train_y)
            neg = len(train_y) - pos
            initial_bias = np.log(pos/neg)
            print(f"The bias is {initial_bias}")
            
            config = model.get_config()
            config["layers"][-1]["config"]["bias_initializer"]["class_name"] = "Constant"
            config["layers"][-1]["config"]["bias_initializer"]["config"]['value'] = initial_bias
            model = model.from_config(config)
            return model
        
        #Set the input shapes of the Dense and RNN networks according to the input data
        def set_input_shapes_DenseRNN(model, baseline_data, fup_data):
            config = model.get_config()
            for i in config['layers']:
                if i["name"] == "baseline_input":
                    i["config"]["batch_input_shape"] =  (None, baseline_data.shape[1]) #This is the baseline_input layer
                elif i["name"] == "RNN_input":
                    i["config"]["batch_input_shape"] =  (None, None, fup_data.shape[2]) #This is the fup_input layer
            
            
            
            # config = model.get_config()
            # config["layers"][0]["config"]["batch_input_shape"] =  (None, baseline_data.shape[1]) #This is the baseline_input layer
            # config["layers"][1]["config"]["batch_input_shape"] =  (None, None, fup_data.shape[2]) #This is the fup_input layer
            # print("Baseline Shape",baseline_data.shape)
            # print("FUP Shape",fup_data.shape)
            # print(model.get_layer)
            # print(config)
            model = model.from_config(config)
            return model
        
        #Get the mean of a cross-validations folds for all the metrics
        def get_mean_median_of_dicts(dicts_dic):
            result_dict = dict()
            metrics_names = list(list(dicts_dic.values())[0].keys())
            
            for metric in metrics_names:
                metric_values_across_folds = [dicts_dic[trial][metric] for trial in dicts_dic]
                result_dict[f"mean_{metric}"] = np.mean(metric_values_across_folds)
                result_dict[f"median_{metric}"] = np.median(metric_values_across_folds)

            return result_dict
        
        #Record history metrics for the cross-validation        
        def record_history_values(history_dict, metric_name, mode, fold_num, cv_results_dict):
            
            #Find the index of the epoch at which the metric is optimized for a given fold cv
            if mode == "min":
                index = np.argmin(history_dict[metric_name])
            elif mode == "max":
                index = np.argmax(history_dict[metric_name])
            
            #Create a new dict which has all the metrics at the optimum epoch for that fold cv
            fold_metric = {key:values[index] for key, values in history_dict.items()}
            
            #Record the result of fold-cv in a dictionary for this trial
            trial_metrics[f"fold_{fold_num}"] = fold_metric
            
            fold_metric_copy = fold_metric.copy()
            
            #Record the result of fold-cv in a dictionary provided to the hypermodel.fit for debugging/graphing
            fold_metric_copy["final_bias"] = model.get_config()["layers"][-1]["config"]["bias_initializer"]["config"]
            fold_metric_copy["best_epoch"] = index + 1
            cv_results_dict[f"trial_{self.trial_number}_fold_{fold_num}"] = fold_metric_copy
        
        #Sampling Technique
        def get_sampling(sampling_method, X, y):
            if sampling_method == "None":
                return X, y
            elif sampling_method == "SMOTE":
                smote = SMOTE()
                return smote.fit_resample(X, y)
            elif sampling_method == "OverSample":
                ovsample = RandomOverSampler()
                return ovsample.fit_resample(X, y)
                
        
        #A dictionary that will store the validation metrics recorded for a given trial
        trial_metrics = dict()
        
        #Get the training values from x
        baseline_train_val_X, fups_train_val_X = x
        
        #Whether to weight the classes or not
        weighted_fit = hp.Boolean("class_weight")
        
        #The choice of learning rate
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        
        if self.model_type == "Dense":
            #The choice of the feature set to use for the baseline dataset
            feature_choice = hp.Choice("feature_set", list(key for key, value in baseline_feature_dic.items()))
            baseline_train_val_X = baseline_train_val_X[baseline_feature_dic[feature_choice]]
            
            print("The baseline length in grid search", baseline_train_val_X.shape)
            
            #Choose a sampling technique
            sampling_method = hp.Choice("data_sampling", ["None", "SMOTE", "OverSample"])
            
        elif self.model_type == "Dense_RNN":
            #The choice of the feature set to use for the baseline dataset
            feature_choice = hp.Choice("feature_set", list(key for key, value in baseline_feature_dic.items()))
            baseline_train_val_X = baseline_train_val_X[baseline_feature_dic[feature_choice]]
        
        if grid_search:
            print("Performing Grid-Search...")

            #Perform 5-fold cross validation
            for fold_num, (training_data, validation_data) in enumerate(kfold_cv(baseline_train_val_X, fups_train_val_X, y, k=5)):
                baseline_train_X, fups_train_X, train_y = training_data
                baseline_valid_X, fups_valid_X, valid_y = validation_data
                
                if self.model_type == "Dense":
                    #Change the sampling method
                    baseline_train_X, train_y = get_sampling(sampling_method, X=baseline_train_X , y=train_y)
                
                #Set the input shape of the RNN and the Dense layer
                if self.model_type == "Dense_RNN":
                    model = set_input_shapes_DenseRNN(model, baseline_train_X, fups_train_X)
                
                
                #Change the bias weight of the final layer
                model = set_output_bias_initial_val(model, train_y)

                #Compile the model
                model.compile(
                    optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                    loss="binary_crossentropy",
                    metrics = self.METRICS,
                )

                
                #Whether to weight the classes or not
                if weighted_fit:
                    class_weight = calculate_class_weight(train_y)
                else:
                    class_weight = {0: 1, 1: 1}

                
                
                #print(fups_train_X.shape, "#$#$##$#$#$")
                if self.model_type == "Dense":
                    #Fit the model with the training and validation sets
                    history = model.fit(
                        x = baseline_train_X,
                        y = train_y,
                        callbacks = [keras.callbacks.EarlyStopping(monitor=f"val_loss", mode="min", patience=5)],
                        validation_data = (baseline_valid_X, valid_y),
                        class_weight = class_weight,
                        **kwargs
                            )
                
                elif self.model_type == "RNN":
                    #Fit the model with the training and validation sets
                    history = model.fit(
                        x = fups_train_X,
                        y = train_y,
                        callbacks = [keras.callbacks.EarlyStopping(monitor=f"val_loss", mode="min", patience=5)],
                        validation_data = (fups_valid_X, valid_y),
                        class_weight = class_weight,
                        **kwargs
                            )
                    
                elif self.model_type == "Dense_RNN":
                    #Fit the model with the training and validation
                    history = model.fit(
                        x = [baseline_train_X, fups_train_X],
                        y = train_y,
                        callbacks = [keras.callbacks.EarlyStopping(monitor=f"val_loss", mode="min", patience=5)],
                        validation_data = ([baseline_valid_X, fups_valid_X], valid_y),
                        class_weight = class_weight,
                        **kwargs
                    )
                
                #Record the data obtained from this fold
                record_history_values(history_dict=history.history,
                                   metric_name=f"val_{metric_name}",
                                   mode = metric_mode,
                                   fold_num = fold_num+1,
                                   cv_results_dict = self.cv_results_dict)
                
            
            #Increment the trial number
            self.trial_number += 1 
            
            #Get the mean of the metrics
            return get_mean_median_of_dicts(trial_metrics)
        
        else:
            print("Performing final training.")
            
            if self.model_type == "Dense":
                #Change the sampling method
                baseline_train_val_X, y = get_sampling(sampling_method, X=baseline_train_val_X , y=y)
            
            
            #Set the input shape of the RNN and the Dense layer
            if self.model_type == "Dense_RNN":
                model = set_input_shapes_DenseRNN(model, baseline_train_val_X, fups_train_val_X )
                
            
            #Change the bias weight of the final layer
            model = set_output_bias_initial_val(model, y)
            
            #Whether to weight the classes or not
            if weighted_fit:
                class_weight = calculate_class_weight(y)
            else:
                class_weight = {0: 1, 1: 1}
            
            #Compile the model
            model.compile(
                optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics = self.METRICS,
            )
            
            if self.model_type == "Dense":
                #Fit the model
                history = model.fit(
                    x = baseline_train_val_X,
                    y = y,
                    class_weight = class_weight,
                    **kwargs,
                            )
            elif self.model_type == "RNN":
                #Fit the model
                history = model.fit(
                    x = fups_train_val_X,
                    y = y,
                    class_weight = class_weight,
                    **kwargs,
                            )
            elif self.model_type == "Dense_RNN":
                #Fit the model
                history = model.fit(
                    x = [baseline_train_val_X, fups_train_val_X],
                    y = y,
                    class_weight = class_weight,
                    **kwargs,
                            )
                

                        
            return history, model

def get_best_number_of_training_epochs(tuner, metric_name, metric_mode):
    
    #Access the results pandas dataframe obtained when tunning
    cv_dic = pd.DataFrame.from_dict(tuner.hypermodel.cv_results_dict, orient='index')
    
    #set of all the trial numbers
    trial_nums = set([int(entry.split("_")[1]) for entry in cv_dic.index])
    
    #Two lists that will contain the mean and median of the metric of interest.
    #median_lists = []
    mean_lists = []

    #For each trial, calculate the mean and median of the metric of interest across all the folds.
    for i in trial_nums:
        #median_ = cv_dic[cv_dic.index.str.startswith(f"trial_{i}_")][metric_name].median()
        mean_ = cv_dic[cv_dic.index.str.startswith(f"trial_{i}_")][metric_name].mean()
        #median_lists.append(median_)
        mean_lists.append(mean_)
        #print(f"Trial {i}, Mean {mean_}, Median {median_}")

    #Using the mean lists, calculate the best trial
    if metric_mode == "max":
        best_trial_num = np.argmax(mean_lists)
        best_metric = np.max(mean_lists)
    elif metric_mode == "min":
        best_trial_num = np.argmin(mean_lists)
        best_metric = np.min(mean_lists)

    #Use the best trial to find the best number of epochs to train the model
    best_number_of_epochs = int(cv_dic[cv_dic.index.str.startswith(f"trial_{i}_")]["best_epoch"].mean())

    print("Best Trial is:", best_trial_num, "With mean metric:", best_metric)
    print("Best number of epochs:", best_number_of_epochs)

    return best_number_of_epochs

def plot_history(history):
    fig, axs = plt.subplots(5, 2, figsize=(15, 20))

    for key, ax in zip([val for val in history.history.keys() if "_" not in val], axs.flat):
        for metric in [f"{key}", f"val_{key}"]:
            ax.plot(range(1, len(history.history[metric])+1), history.history[metric],"-o", label=metric)
            ax.legend()
            ax.set_xlabel("Epochs")