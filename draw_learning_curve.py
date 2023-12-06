from data_preparation import get_formatted_Baseline_FUP
from hypermodel_experiments import get_predefined_training_testing_indeces_30_percent, get_X_y_from_indeces, create_feature_set_dicts_baseline_and_FUP
from constants import NUMBER_OF_ITERATIONS_CV, timeseries_padding_value
from cross_validation import kfold_cv
import numpy as np
from hypermodel_helper import calculate_class_weight
import tensorflow as tf
import matplotlib.pyplot as plt
import copy
from hypermodels import get_last_FUPs_array


def get_training_val_curve_Baseline():
    patient_dataset, FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = get_formatted_Baseline_FUP(mode="Formatted")

    print(f"Follow-up data has {len(FUPS_dict)} examples and {len(list_FUP_cols)} features.")
    print(f"Baseline data has {len(baseline_dataframe)} examples and {len(baseline_dataframe.columns)} features.")
    print(f"The target data has {len(target_series)} data.", "\n")

    experiment_name = "Baseline_Dense"

    metric_name = "prc"
    metric_mode = "max"

    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()


    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset, mode="only Baseline")

    #The choice of the feature set to use for the baseline dataset
    baseline_feature_choice = "20_f_classif"
    features_index_to_keep_baseline = feature_selection_dict["baseline_feature_sets"][baseline_feature_choice]
    baseline_train_val_X = baseline_train_val_X.iloc[:,features_index_to_keep_baseline]
    
    history_dics = dict()
    hist_counter = 1
    
    for repeat in range(1, NUMBER_OF_ITERATIONS_CV+1):
        #Perform 2-fold cross validation
        for fold_num, (training_data, validation_data) in enumerate(kfold_cv(baseline_train_val_X, fups_train_val_X, train_val_y, k=2, timeseries_padding_value=timeseries_padding_value, deterministic=False)):
            baseline_train_X, _, train_y = training_data
            baseline_valid_X, _, valid_y = validation_data
            
            #Bias value of the output node
            pos = sum(train_y)
            neg = len(train_y) - pos
            initial_bias = np.log(pos/neg)
                        
            #Reset the model by intializing it again. Otherwise, we will be training the same model over and over across each fold/iteration.
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(units=40, name= f"Dense_{0}",activation = 'relu',kernel_regularizer = None))
            model.add(tf.keras.layers.Dropout(rate=0.25, name= f"Dropout_{0}"))
            model.add(tf.keras.layers.Dense(units=40, name= f"Dense_{1}",activation = 'relu',kernel_regularizer = None))
            model.add(tf.keras.layers.Dropout(rate=0.25, name= f"Dropout_{1}"))
            model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid", bias_initializer = tf.keras.initializers.Constant(value=initial_bias)))
            

            #Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0005),
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
                batch_size=16,
                verbose = 0,
                epochs=100,
                )
            
            
            
            history_dics[f'trial_{hist_counter}']=history.history
            
            hist_counter += 1
            
    plot_training_val_curve(history_dics, "Baseline_dense")
            
    
def get_training_val_curve_LastFUP():
    patient_dataset, FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = get_formatted_Baseline_FUP(mode="Formatted")

    print(f"Follow-up data has {len(FUPS_dict)} examples and {len(list_FUP_cols)} features.")
    print(f"Baseline data has {len(baseline_dataframe)} examples and {len(baseline_dataframe.columns)} features.")
    print(f"The target data has {len(target_series)} data.", "\n")

    metric_name = "prc"
    metric_mode = "max"

    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()


    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset, mode="only FUP")

    #The choice of the feature set to use for the baseline dataset
    patient_dataset_train_val = copy.deepcopy(patient_dataset)    #Keep only the train_val data for feature selection of the FUP data
    patient_dataset_train_val.all_patients = [patient for patient in patient_dataset_train_val.all_patients if patient.uniqid in training_val_indeces]
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset_train_val, mode="only FUP")
    
    
    # baseline_feature_choice = "20_f_classif"
    # features_index_to_keep_baseline = feature_selection_dict["baseline_feature_sets"][baseline_feature_choice]
    # baseline_train_val_X = baseline_train_val_X.iloc[:,features_index_to_keep_baseline]
    
    fups_feature_choice = 'all_features'
    features_index_to_keep_fups = feature_selection_dict["FUPS_feature_sets"][fups_feature_choice]
    
    history_dics = dict()
    hist_counter = 1
    
    for repeat in range(1, NUMBER_OF_ITERATIONS_CV+1):
        #Perform 2-fold cross validation
        for fold_num, (training_data, validation_data) in enumerate(kfold_cv(baseline_train_val_X, fups_train_val_X, train_val_y, k=2, timeseries_padding_value=timeseries_padding_value, deterministic=False)):
            _, fups_train_X, train_y = training_data
            _, fups_valid_X, valid_y = validation_data
            
            
            #Only use the appropriate feature set
            fups_train_X = fups_train_X[:,:,features_index_to_keep_fups]
            fups_valid_X = fups_valid_X[:,:,features_index_to_keep_fups]
            
            #Only keep the final FUP vector
            fups_train_X = get_last_FUPs_array(fups_X=fups_train_X, timeseries_padding_value=timeseries_padding_value)
            fups_valid_X = get_last_FUPs_array(fups_X=fups_valid_X, timeseries_padding_value=timeseries_padding_value)
            
            
            #Bias value of the output node
            pos = sum(train_y)
            neg = len(train_y) - pos
            initial_bias = np.log(pos/neg)
                        
            #Reset the model by intializing it again. Otherwise, we will be training the same model over and over across each fold/iteration.
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(units=90, name= f"Dense_{0}",activation = 'relu',kernel_regularizer = None))
            model.add(tf.keras.layers.Dropout(rate=0.50, name= f"Dropout_{0}"))
            model.add(tf.keras.layers.Dense(units=90, name= f"Dense_{1}",activation = 'relu',kernel_regularizer = None))
            model.add(tf.keras.layers.Dropout(rate=0.50, name= f"Dropout_{1}"))
            model.add(tf.keras.layers.Dense(units=90, name= f"Dense_{2}",activation = 'relu',kernel_regularizer = None))
            model.add(tf.keras.layers.Dropout(rate=0.50, name= f"Dropout_{2}"))
            model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid", bias_initializer = tf.keras.initializers.Constant(value=initial_bias)))
            

            #Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0005),
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
                x = fups_train_X,
                y = train_y,
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor=f"val_{metric_name}", mode=metric_mode, patience=10)],
                validation_data = (fups_valid_X, valid_y),
                class_weight = class_weight,
                batch_size=16,
                verbose = 0,
                epochs=100,
                )
            
            
            
            history_dics[f'trial_{hist_counter}']=history.history
            
            hist_counter += 1
            
    plot_training_val_curve(history_dics, "Last_FUP_Dense")           


def get_training_val_curve_FUP_RNN():
    patient_dataset, FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = get_formatted_Baseline_FUP(mode="Formatted")

    print(f"Follow-up data has {len(FUPS_dict)} examples and {len(list_FUP_cols)} features.")
    print(f"Baseline data has {len(baseline_dataframe)} examples and {len(baseline_dataframe.columns)} features.")
    print(f"The target data has {len(target_series)} data.", "\n")

    metric_name = "prc"
    metric_mode = "max"

    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()


    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset, mode="only FUP")

    #The choice of the feature set to use for the baseline dataset
    patient_dataset_train_val = copy.deepcopy(patient_dataset)    #Keep only the train_val data for feature selection of the FUP data
    patient_dataset_train_val.all_patients = [patient for patient in patient_dataset_train_val.all_patients if patient.uniqid in training_val_indeces]
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset_train_val, mode="only FUP")
    
    fups_feature_choice = 'all_features'
    features_index_to_keep_fups = feature_selection_dict["FUPS_feature_sets"][fups_feature_choice]
    
    history_dics = dict()
    hist_counter = 1
    
    for repeat in range(1, NUMBER_OF_ITERATIONS_CV+1):
        #Perform 2-fold cross validation
        for fold_num, (training_data, validation_data) in enumerate(kfold_cv(baseline_train_val_X, fups_train_val_X, train_val_y, k=2, timeseries_padding_value=timeseries_padding_value, deterministic=False)):
            _, fups_train_X, train_y = training_data
            _, fups_valid_X, valid_y = validation_data
            
            
            #Only use the appropriate feature set
            fups_train_X = fups_train_X[:,:,features_index_to_keep_fups]
            fups_valid_X = fups_valid_X[:,:,features_index_to_keep_fups]
            
            #Bias value of the output node
            pos = sum(train_y)
            neg = len(train_y) - pos
            initial_bias = np.log(pos/neg)
                        
            #Reset the model by intializing it again. Otherwise, we will be training the same model over and over across each fold/iteration.
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Masking(mask_value=timeseries_padding_value))
            model.add(tf.keras.layers.LSTM(units=50, name="LSTM_0", recurrent_dropout = 0.25, 
                                           activation = 'tanh',
                                           kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
            
            model.add(tf.keras.layers.Dropout(rate=0.25, name="final_dropout_layer"))
            model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid", bias_initializer = tf.keras.initializers.Constant(value=initial_bias)))
            

            #Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
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
                x = fups_train_X,
                y = train_y,
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor=f"val_{metric_name}", mode=metric_mode, patience=10)],
                validation_data = (fups_valid_X, valid_y),
                class_weight = class_weight,
                batch_size=16,
                verbose = 0,
                epochs=50,
                )
            
            
            
            history_dics[f'trial_{hist_counter}']=history.history
            
            hist_counter += 1
            
    plot_training_val_curve(history_dics, "FUP_RNN")           




def plot_training_val_curve(history_dict, name):
    colors = plt.cm.tab20(range(10))

    fig, axs = plt.subplots(2, 1, figsize=(10, 7))

    for trial_num, col in zip(range(1, 11), colors):
        axs[0].plot(range(1, len(history_dict[f'trial_{trial_num}']['loss'])+1), history_dict[f'trial_{trial_num}']['loss'],"-", label=f'{trial_num}_loss', c=col)
        axs[0].plot(range(1, len(history_dict[f'trial_{trial_num}']['val_loss'])+1), history_dict[f'trial_{trial_num}']['val_loss'],"--", label=f'{trial_num}_val_loss', c=col)
        axs[0].set_ylabel("Loss")
        
    for trial_num, col in zip(range(1, 11), colors):
        axs[1].plot(range(1, len(history_dict[f'trial_{trial_num}']['loss'])+1), history_dict[f'trial_{trial_num}']['prc'],"-", label=f'{trial_num}_train', c=col)
        axs[1].plot(range(1, len(history_dict[f'trial_{trial_num}']['val_loss'])+1), history_dict[f'trial_{trial_num}']['val_prc'],"--", label=f'{trial_num}_val', c=col)
        axs[1].legend(loc="center left", bbox_to_anchor=(1.04, 1))
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("AUPRC")
        
    axs[0].set_title(name)
    
    plt.savefig(f"./diagnosis/{name}.pdf", transparent=False, bbox_inches="tight") 


if __name__ == "__main__":
    
    # get_training_val_curve_Baseline()
    # get_training_val_curve_LastFUP()
    get_training_val_curve_FUP_RNN()
    