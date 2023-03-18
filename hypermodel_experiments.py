from rnn import LastFUPHyperModel, BaselineHyperModel, FUP_RNN_HyperModel, Baseline_FUP_Multiinput_HyperModel
import keras_tuner
import pandas as pd
from cross_validation import divide_into_stratified_fractions, get_X_y_from_indeces, normalize_training_validation
from sklearn.feature_selection import GenericUnivariateSelect, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import numpy as np
from constants import timeseries_padding_value
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")


def get_important_feature_dic_baseline(baseline_dataset, target_list):
    #This function is for feature selection
    def get_important_features(X, y, method, number_features):
        transformer = GenericUnivariateSelect(method, mode='k_best', param=number_features)
        X_new = transformer.fit(X, y)
        return X_new.get_feature_names_out()
    
    
    features_dict = dict()
    methods_list = [f_classif, mutual_info_classif]
    numbers_list = [1, 5, 10, 15, 20, 30]

    for method in methods_list:
        for number in numbers_list:
            features_dict[f"{number}_{method.__name__}"] = get_important_features(baseline_dataset, 
                                                                        target_list, 
                                                                        method, 
                                                                        number)

    features_dict["total"] = baseline_dataset.columns
    
    return features_dict

def create_feature_set_dicts_baseline_and_FUP(baseline_dataframe, FUP_columns, target_series):
    """Perform feature selection for the baseline data and FUP, then returns an appropriate dict for hyperparameter tuning.

    Args:
        baseline_dataframe (pd.dataframe): The baseline data (that come from training-val data) SHOULD NOT be standardized.
        FUP_columns (list): List of FUP feature names.
        target_series (pd.series): Target varaibles for the patients.

    Returns:
        dict: Dicts with keys "baseline_feature_sets" and "FUPS_feature_sets" containing the features sets for baseline and FUP.
    """
    #Standardize the baseline dataframe
    baseline_dataframe_standardized = pd.DataFrame(StandardScaler().fit_transform(baseline_dataframe.copy()), columns = baseline_dataframe.columns)

    #Perform feature selection with different number of features
    baseline_feature_sets_dict = get_important_feature_dic_baseline(baseline_dataframe_standardized, target_series)

    #Convert the names to index for the values in the baseline_feature_sets_dict
    indexed_feature_set_dict = dict()
    all_cols = list(baseline_dataframe.columns)

    for feature_set in baseline_feature_sets_dict:
        indexed_feature = [all_cols.index(feature) for feature in baseline_feature_sets_dict[feature_set]]
        indexed_feature_set_dict[feature_set] = indexed_feature
        
    ####################
    ###################

    #Two sets of features in FUP data

    FUP_features_sets = dict()

    FUP_features_sets["total"] = list(range(len(FUP_columns)))

    features_without_new_no_continue_yes = [feature for feature in FUP_columns if (feature.find("_Yes")==-1) & (feature.find("_No")==-1) & \
                                            (feature.find("_Continue")==-1) & (feature.find("_New")==-1)]


    FUP_features_sets["FUP_without_new_no_continue_yes"] = [FUP_columns.index(feature) for feature in features_without_new_no_continue_yes]

    return {"baseline_feature_sets": indexed_feature_set_dict, "FUPS_feature_sets":FUP_features_sets}

def run_baseline_dense_experiment(model_name, 
                                 directory_name, 
                                 metric_name, 
                                 metric_mode, 
                                 metric_cv_calc_mode, 
                                 baseline_dataframe,
                                 FUPS_dict,
                                 target_series,
                                 list_FUP_cols,
                                 overwrite=False
                                 ):
    
    """Performs hyperparamter search, save the best model, and then test on the testing set for the lastFUP_dense experiment.

    Args:
        model_name (string): The name of the hypermodel (could be anything).
        directory_name (string): The name of the directory to save the models info.
        metric_name (string): The name of the metric to optimize in hypermodel.
        metric_mode (string): Either optimize based on the "min" or "max" of the metric.
        metric_cv_calc_mode (string): Either "mean" or "median" to use for selecting the best hyperparameter.
        baseline_dataframe (pandas.DataFrame): The Baseline dataset.
        FUPS_dict (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        target_series (pandas.Series): Pandas Series of all the patients labels.
        list_FUP_cols (list): The list of the feature names (in order) for the FUP columns.
        overwrite (bool): Whether to overwrite the keras-tuner results directory.
    """
    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    #Record the training_val and testing indeces
    record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)

    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y)
    del feature_selection_dict["FUPS_feature_sets"] #Remove the FUPs feature sets becuase they aren't used here.
    
    #Define the tuner
    tuner = keras_tuner.RandomSearch(
                    BaselineHyperModel(name = model_name),
                    objective=keras_tuner.Objective(f"{metric_cv_calc_mode}_val_{metric_name}", metric_mode),
                    max_trials=20,
                    seed=1375,
                    overwrite = overwrite,
                    directory=directory_name,
                    project_name=model_name)

    #Perform the search
    tuner.search(
                x = [baseline_train_val_X, fups_train_val_X], 
                y = train_val_y,
                batch_size = 16,
                grid_search = True,
                epochs = 100,
                metric_name = metric_name,
                metric_mode = metric_mode,
                verbose = 0,
                feature_set_dict=feature_selection_dict
                )
    
    #Final training on the training-val 
    best_hp = tuner.get_best_hyperparameters()[0] #Get the best hyperparameters
    print("\n",best_hp.values)
    
    #A dict to store hp values
    best_hp_values_dict = {hp:value for hp, value in best_hp.values.items()}
    
    #Calculate the best number of epochs to train the data
    best_number_of_epochs = get_best_number_of_training_epochs(tuner, 
                                                               metric_name=f"val_{metric_name}", 
                                                               metric_mode=metric_mode, 
                                                               metric_cv_calc_mode=metric_cv_calc_mode)
    
    #Save the best hps and best number of epochs in a csv.
    best_hp_values_dict["best_number_of_epochs"] = best_number_of_epochs
    pd.DataFrame.from_dict(best_hp_values_dict, orient="index", columns=["best_hp_value"]).to_csv(f"{directory_name}/{model_name}_best_hp.csv")


    #################
    #####Training on the entire train_val dataset
    #################
    
    #Standardize the training data, and the test data.
    norm_train_val_data , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    norm_train_val_baseline_X, norm_train_val_fups_X, train_val_y = norm_train_val_data
    norm_test_baseline_X, _, test_y = norm_test_data

    #Train the model on the entire train_val dataset with best hyperparameters for best # of epochs
    hypermodel = BaselineHyperModel(name = model_name)
    model_keras_tuner = tuner.hypermodel.build(best_hp)
    history, model_keras_tuner = hypermodel.fit(best_hp, model_keras_tuner, 
                                    [norm_train_val_baseline_X, norm_train_val_fups_X], 
                                    train_val_y, 
                                    grid_search=False, 
                                    batch_size = 16,
                                    epochs=int(best_number_of_epochs),
                                    feature_set_dict=feature_selection_dict)
    
    #Save the trained model
    model_keras_tuner.save(f"{directory_name}/{model_name}.h5")
    
    #Load the saved model
    model = keras.models.load_model(f"{directory_name}/{model_name}.h5")
    
    #Only keep the best features (as determined by best_hp)
    #Note that we didn't need to do this when calling fit, because it is done internally in the class.
    #However, here it should be done before testing the dataset.
    print(f"Using the feature set {best_hp.values['baseline_feature_set']} for testing.")
    norm_test_baseline_X = norm_test_baseline_X.iloc[:,feature_selection_dict["baseline_feature_sets"][best_hp.values["baseline_feature_set"]]]
    norm_train_val_baseline_X = norm_train_val_baseline_X.iloc[:,feature_selection_dict["baseline_feature_sets"][best_hp.values["baseline_feature_set"]]]
    
    #Test on the testing set
    test_res = model.evaluate(norm_test_baseline_X, test_y)
    pd.DataFrame.from_dict({metric:value for metric, value in zip(model_keras_tuner.metrics_names, test_res)}, orient="index", columns=["test"]).to_csv(f"{directory_name}/{model_name}_test_result.csv")

    #Test on the testing set and training_val set
    #Note, we are testing on the test set with the same model twice (previous 2 lines) which is just for debugging.
    save_metrics_and_ROC_PR(model_name=model_name, 
                            model=model, 
                            training_x=norm_train_val_baseline_X, 
                            training_y=train_val_y, 
                            testing_x = norm_test_baseline_X, 
                            testing_y = test_y)

def run_lastFUP_dense_experiment(model_name, 
                                 directory_name, 
                                 metric_name, 
                                 metric_mode, 
                                 metric_cv_calc_mode, 
                                 baseline_dataframe,
                                 FUPS_dict,
                                 target_series,
                                 list_FUP_cols,
                                 overwrite=False
                                 ):
    
    """Performs hyperparamter search, save the best model, and then test on the testing set for the lastFUP_dense experiment.

    Args:
        model_name (string): The name of the hypermodel (could be anything).
        directory_name (string): The name of the directory to save the models info.
        metric_name (string): The name of the metric to optimize in hypermodel.
        metric_mode (string): Either optimize based on the "min" or "max" of the metric.
        metric_cv_calc_mode (string): Either "mean" or "median" to use for selecting the best hyperparameter.
        baseline_dataframe (pandas.DataFrame): The Baseline dataset.
        FUPS_dict (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        target_series (pandas.Series): Pandas Series of all the patients labels.
        list_FUP_cols (list): The list of the feature names (in order) for the FUP columns.
        overwrite (bool): Whether to overwrite the keras-tuner results directory.
    """
    
    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)

    #record the indeces of the train_val and test dataset
    record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)
    
    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y)
    del feature_selection_dict["baseline_feature_sets"] #Remove the baseline feature sets becuase they aren't used here.
    
    #Define the tuner
    tuner = keras_tuner.RandomSearch(
                    LastFUPHyperModel(name = model_name),
                    objective=keras_tuner.Objective(f"{metric_cv_calc_mode}_val_{metric_name}", metric_mode),
                    max_trials=20,
                    seed=1375,
                    overwrite = overwrite,
                    directory=directory_name,
                    project_name=model_name)

    #Perform the search
    tuner.search(
                x = [baseline_train_val_X, fups_train_val_X], 
                y = train_val_y,
                batch_size = 16,
                grid_search = True,
                epochs = 50,
                metric_name = metric_name,
                metric_mode = metric_mode,
                verbose = 0,
                feature_set_dict=feature_selection_dict
                )
    
    #Final training on the training-val 
    best_hp = tuner.get_best_hyperparameters()[0] #Get the best hyperparameters
    print("\n",best_hp.values)
    
    best_hp_values_dict = {hp:value for hp, value in best_hp.values.items()}
    #Calculate the best number of epochs to train the data
    best_number_of_epochs = get_best_number_of_training_epochs(tuner, 
                                                               metric_name=f"val_{metric_name}", 
                                                               metric_mode=metric_mode, 
                                                               metric_cv_calc_mode=metric_cv_calc_mode)
    
    #Save the best hps and best number of epochs in a csv.
    best_hp_values_dict["best_number_of_epochs"] = best_number_of_epochs
    pd.DataFrame.from_dict(best_hp_values_dict, orient="index", columns=["best_hp_value"]).to_csv(f"{directory_name}/{model_name}_best_hp.csv")

    #################
    #####Training on the entire train_val dataset using best hps
    #################
    
    #Standardize the training data, and the test data.
    norm_train_val_data , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    norm_train_val_baseline_X, norm_train_val_fups_X, train_val_y = norm_train_val_data
    _                        , norm_test_fups_X, test_y = norm_test_data

    #Train the model on the entire train_val dataset with best hyperparameters for best # of epochs
    hypermodel = LastFUPHyperModel(name = model_name)
    model_keras_tuner = tuner.hypermodel.build(best_hp)
    history, model_keras_tuner = hypermodel.fit(best_hp, model_keras_tuner, 
                                    [norm_train_val_baseline_X, norm_train_val_fups_X], 
                                    train_val_y, 
                                    grid_search=False, 
                                    batch_size = 16,
                                    epochs=int(best_number_of_epochs),
                                    feature_set_dict=feature_selection_dict)
    
    #Save the trained model
    model_keras_tuner.save(f"{directory_name}/{model_name}.h5")
    
    #Load the saved model
    model = keras.models.load_model(f"{directory_name}/{model_name}.h5")
    
    #Extract the last FUP for each patient in the test set.
    #Note that we didn't need to do this when calling fit, because it is done internally.
    #However, here it is not done internally.
    print(f"Using the feature set {best_hp.values['FUPS_feature_set']} for testing.")
    norm_test_fups_X_last = get_last_FUPs_array(norm_test_fups_X, timeseries_padding_value)
    norm_test_fups_X_last = norm_test_fups_X_last[:,feature_selection_dict["FUPS_feature_sets"][best_hp.values["FUPS_feature_set"]]]
    
    #Do the same for training_val and testing data
    norm_training_val_fups_X_last = get_last_FUPs_array(norm_train_val_fups_X, timeseries_padding_value)
    norm_training_val_fups_X_last = norm_training_val_fups_X_last[:,feature_selection_dict["FUPS_feature_sets"][best_hp.values["FUPS_feature_set"]]]
    
    #Test on the testing set
    test_res = model.evaluate(norm_test_fups_X_last, test_y)
    pd.DataFrame.from_dict({metric:value for metric, value in zip(model_keras_tuner.metrics_names, test_res)}, orient="index", columns=["test"]).to_csv(f"{directory_name}/{model_name}_test_result.csv")

    save_metrics_and_ROC_PR(model_name=model_name, 
                            model=model, 
                            training_x=norm_training_val_fups_X_last, 
                            training_y=train_val_y, 
                            testing_x = norm_test_fups_X_last, 
                            testing_y = test_y)

def run_FUP_RNN_experiment(model_name, 
                                 directory_name, 
                                 metric_name, 
                                 metric_mode, 
                                 metric_cv_calc_mode, 
                                 baseline_dataframe,
                                 FUPS_dict,
                                 target_series,
                                 list_FUP_cols,
                                 overwrite=False
                                 ):
    
    """Performs hyperparamter search, save the best model, and then test on the testing set for the lastFUP_dense experiment.

    Args:
        model_name (string): The name of the hypermodel (could be anything).
        directory_name (string): The name of the directory to save the models info.
        metric_name (string): The name of the metric to optimize in hypermodel.
        metric_mode (string): Either optimize based on the "min" or "max" of the metric.
        metric_cv_calc_mode (string): Either "mean" or "median" to use for selecting the best hyperparameter.
        baseline_dataframe (pandas.DataFrame): The Baseline dataset.
        FUPS_dict (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        target_series (pandas.Series): Pandas Series of all the patients labels.
        list_FUP_cols (list): The list of the feature names (in order) for the FUP columns.
        overwrite (bool): Whether to overwrite the keras-tuner results directory.
    """
    
    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)

    #Record the training_val and testing indeces
    record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)
    
    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y)
    del feature_selection_dict["baseline_feature_sets"] #Remove the baseline feature sets becuase they aren't used here.
    
    #Define the tuner
    tuner = keras_tuner.RandomSearch(
                    FUP_RNN_HyperModel(name = model_name),
                    objective=keras_tuner.Objective(f"{metric_cv_calc_mode}_val_{metric_name}", metric_mode),
                    max_trials=20,
                    seed=1375,
                    overwrite = overwrite,
                    directory=directory_name,
                    project_name=model_name)

    #Perform the search
    tuner.search(
                x = [baseline_train_val_X, fups_train_val_X], 
                y = train_val_y,
                batch_size = 16,
                grid_search = True,
                epochs = 50,
                metric_name = metric_name,
                metric_mode = metric_mode,
                verbose = 1,
                feature_set_dict=feature_selection_dict
                )
    
    #Final training on the training-val 
    best_hp = tuner.get_best_hyperparameters()[0] #Get the best hyperparameters
    print("\n",best_hp.values, "\n")
    
    best_hp_values_dict = {hp:value for hp, value in best_hp.values.items()}
    #Calculate the best number of epochs to train the data
    best_number_of_epochs = get_best_number_of_training_epochs(tuner, 
                                                               metric_name=f"val_{metric_name}", 
                                                               metric_mode=metric_mode, 
                                                               metric_cv_calc_mode=metric_cv_calc_mode)
    
    #Save the best hps and best number of epochs in a csv.
    best_hp_values_dict["best_number_of_epochs"] = best_number_of_epochs
    pd.DataFrame.from_dict(best_hp_values_dict, orient="index", columns=["best_hp_value"]).to_csv(f"{directory_name}/{model_name}_best_hp.csv")

    #################
    #####Training on the entire train_val dataset using best hps
    #################
    
    #Standardize the training data, and the test data.
    norm_train_val_data , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    norm_train_val_baseline_X, norm_train_val_fups_X, train_val_y = norm_train_val_data
    _                        , norm_test_fups_X, test_y = norm_test_data

    #Train the model on the entire train_val dataset with best hyperparameters for best # of epochs
    hypermodel = FUP_RNN_HyperModel(name = model_name)
    model_keras_tuner = tuner.hypermodel.build(best_hp)
    history, model_keras_tuner = hypermodel.fit(best_hp, model_keras_tuner, 
                                    [norm_train_val_baseline_X, norm_train_val_fups_X], 
                                    train_val_y, 
                                    grid_search=False, 
                                    batch_size = 16,
                                    epochs=int(best_number_of_epochs),
                                    feature_set_dict=feature_selection_dict)
    
    #Save the trained model
    model_keras_tuner.save(f"{directory_name}/{model_name}.h5")
    
    #Load the saved model
    model = keras.models.load_model(f"{directory_name}/{model_name}.h5")
    
    #Extract the last FUP for each patient in the test set.
    #Note that we didn't need to do this when calling fit, because it is done internally.
    #However, here it is not done internally.
    print(f"Using the feature set {best_hp.values['FUPS_feature_set']} for testing.")
    norm_test_fups_X = norm_test_fups_X[:,:,feature_selection_dict["FUPS_feature_sets"][best_hp.values["FUPS_feature_set"]]]
    
    #Prepare the training-val dataset for the model (like the test data)
    norm_train_val_fups_X = norm_train_val_fups_X[:,:,feature_selection_dict["FUPS_feature_sets"][best_hp.values["FUPS_feature_set"]]]    
    
    #Test on the testing set
    test_res = model.evaluate(norm_test_fups_X, test_y)
    pd.DataFrame.from_dict({metric:value for metric, value in zip(model_keras_tuner.metrics_names, test_res)}, orient="index", columns=["test"]).to_csv(f"{directory_name}/{model_name}_test_result.csv")

    save_metrics_and_ROC_PR(model_name=model_name, 
                            model=model, 
                            training_x=norm_train_val_fups_X, 
                            training_y=train_val_y, 
                            testing_x = norm_test_fups_X, 
                            testing_y = test_y)

def run_Baseline_FUP_multiinput_experiment(model_name, 
                                 directory_name, 
                                 metric_name, 
                                 metric_mode, 
                                 metric_cv_calc_mode, 
                                 baseline_dataframe,
                                 FUPS_dict,
                                 target_series,
                                 list_FUP_cols,
                                 overwrite=False
                                 ):
    
    """Performs hyperparamter search, save the best model, and then test on the testing set for the lastFUP_dense experiment.

    Args:
        model_name (string): The name of the hypermodel (could be anything).
        directory_name (string): The name of the directory to save the models info.
        metric_name (string): The name of the metric to optimize in hypermodel.
        metric_mode (string): Either optimize based on the "min" or "max" of the metric.
        metric_cv_calc_mode (string): Either "mean" or "median" to use for selecting the best hyperparameter.
        baseline_dataframe (pandas.DataFrame): The Baseline dataset.
        FUPS_dict (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        target_series (pandas.Series): Pandas Series of all the patients labels.
        list_FUP_cols (list): The list of the feature names (in order) for the FUP columns.
        overwrite (bool): Whether to overwrite the keras-tuner results directory.
    """
    
    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)

    #Record the training and testing indeces
    record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)
    
    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y)
    
    #Define the tuner
    tuner = keras_tuner.RandomSearch(
                    Baseline_FUP_Multiinput_HyperModel(name = model_name),
                    objective=keras_tuner.Objective(f"{metric_cv_calc_mode}_val_{metric_name}", metric_mode),
                    max_trials=30,
                    seed=1375,
                    overwrite = overwrite,
                    directory=directory_name,
                    project_name=model_name)

    #Perform the search
    tuner.search(
                x = [baseline_train_val_X, fups_train_val_X], 
                y = train_val_y,
                batch_size = 16,
                grid_search = True,
                epochs = 50,
                metric_name = metric_name,
                metric_mode = metric_mode,
                verbose = 0,
                feature_set_dict=feature_selection_dict
                )
    
    #Final training on the training-val 
    best_hp = tuner.get_best_hyperparameters()[0] #Get the best hyperparameters
    print("\n",best_hp.values)
    
    best_hp_values_dict = {hp:value for hp, value in best_hp.values.items()}
    #Calculate the best number of epochs to train the data
    best_number_of_epochs = get_best_number_of_training_epochs(tuner, 
                                                               metric_name=f"val_{metric_name}", 
                                                               metric_mode=metric_mode, 
                                                               metric_cv_calc_mode=metric_cv_calc_mode)
    
    #Save the best hps and best number of epochs in a csv.
    best_hp_values_dict["best_number_of_epochs"] = best_number_of_epochs
    pd.DataFrame.from_dict(best_hp_values_dict, orient="index", columns=["best_hp_value"]).to_csv(f"{directory_name}/{model_name}_best_hp.csv")

    #################
    #####Training on the entire train_val dataset using best hps
    #################
    
    #Standardize the training data, and the test data.
    norm_train_val_data , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    norm_train_val_baseline_X, norm_train_val_fups_X, train_val_y = norm_train_val_data
    norm_test_baseline_X, norm_test_fups_X, test_y = norm_test_data

    #Train the model on the entire train_val dataset with best hyperparameters for best # of epochs
    hypermodel = Baseline_FUP_Multiinput_HyperModel(name = model_name)
    model_keras_tuner = tuner.hypermodel.build(best_hp)
    history, model_keras_tuner = hypermodel.fit(best_hp, model_keras_tuner, 
                                    [norm_train_val_baseline_X, norm_train_val_fups_X], 
                                    train_val_y, 
                                    grid_search=False, 
                                    batch_size = 16,
                                    epochs=int(best_number_of_epochs),
                                    feature_set_dict=feature_selection_dict)
    
    #Save the trained model
    model_keras_tuner.save(f"{directory_name}/{model_name}.h5")
    
    #Load the saved model
    model = keras.models.load_model(f"{directory_name}/{model_name}.h5")
    
    #"baseline_feature_sets" and "FUPS_feature_sets"
    #The choice of the feature set to use for the baseline dataset
    baseline_feature_choice = best_hp.values["baseline_feature_set"]
    features_index_to_keep_baseline = feature_selection_dict["baseline_feature_sets"][baseline_feature_choice]
    norm_train_val_baseline_X = norm_train_val_baseline_X.iloc[:,features_index_to_keep_baseline]
    norm_test_baseline_X = norm_test_baseline_X.iloc[:,features_index_to_keep_baseline]
    
    #The choice of the feature set to use for FUP data
    fups_feature_choice = best_hp.values["FUPS_feature_set"]
    features_index_to_keep_fups = feature_selection_dict["FUPS_feature_sets"][fups_feature_choice]
    norm_test_fups_X = norm_test_fups_X[:,:,features_index_to_keep_fups]
    norm_train_val_fups_X = norm_train_val_fups_X[:,:,features_index_to_keep_fups]
    
    #Note that we didn't need to do this when calling fit, because it is done internally.
    #However, here it is not done internally.
    print(f"Using the baseline feature set {baseline_feature_choice} and FUP feature set {fups_feature_choice} for testing.")

    
    #Test on the testing set
    test_res = model.evaluate([norm_test_baseline_X, norm_test_fups_X], test_y)
    pd.DataFrame.from_dict({metric:value for metric, value in zip(model_keras_tuner.metrics_names, test_res)}, orient="index", columns=["test"]).to_csv(f"{directory_name}/{model_name}_test_result.csv")

    save_metrics_and_ROC_PR(model_name=model_name, 
                            model=model, 
                            training_x=[norm_train_val_baseline_X, norm_train_val_fups_X], 
                            training_y=train_val_y, 
                            testing_x = [norm_test_baseline_X, norm_test_fups_X], 
                            testing_y = test_y)

def get_best_number_of_training_epochs(tuner, metric_name, metric_mode, metric_cv_calc_mode):
    """Calculates the best number of epochs to train the model for.

    Args:
        tuner (keras_tuner.tuner): A tuned keras tuner object.
        metric_name (string): The name of the metric we are trying to optimize.
        metric_mode (string): Either optimize based on the "min" or "max" of the metric.
        metric_cv_calc_mode (string): Either "mean" or "median" to use for selecting the best hyperparameter.

    Returns:
        int: Best number of epochs to train the model. 
    """
    
    #Access the results pandas dataframe obtained when tunning
    cv_dic = pd.DataFrame.from_dict(tuner.hypermodel.cv_results_dict, orient='index')
    
    #set of all the trial numbers
    trial_nums = list(set([int(entry.split("_")[1]) for entry in cv_dic.index]))
    
    #Two lists that will contain the mean and median of the metric of interest.
    #
    median_lists = []
    mean_lists = []

    #For each trial, calculate the mean and median of the metric of interest across all the folds.
    for i in trial_nums:
        median_ = cv_dic[cv_dic.index.str.startswith(f"trial_{i}_")][metric_name].median()
        mean_ = cv_dic[cv_dic.index.str.startswith(f"trial_{i}_")][metric_name].mean()
        median_lists.append(median_)
        mean_lists.append(mean_)
        #print(f"Trial {i}, Mean {mean_}, Median {median_}")

    #Using the mean lists, calculate the best trial
    
    if metric_mode == "max":
        best_trial_num_by_mean = trial_nums[np.argmax(mean_lists)]
        best_trial_num_by_median = trial_nums[np.argmax(median_lists)]
        best_metric_by_mean = np.max(mean_lists)
        best_metric_by_median = np.max(median_lists)
    elif metric_mode == "min":
        best_trial_num_by_mean = trial_nums[np.argmin(mean_lists)]
        best_trial_num_by_median = trial_nums[np.argmin(median_lists)]
        best_metric_by_mean = np.min(mean_lists)
        best_metric_by_median = np.min(median_lists)

    #Use the best trial to find the best number of epochs to train the model
    if metric_cv_calc_mode == "mean":
        best_number_of_epochs = round(cv_dic[cv_dic.index.str.startswith(f"trial_{best_trial_num_by_mean}_")]["best_epoch"].mean())
        print("Best Trial by mean is:", best_trial_num_by_mean, f"with {metric_name}: {best_metric_by_mean}") 
        print("Best number of epochs by mean is:", best_number_of_epochs)
        return best_number_of_epochs
    
    elif metric_cv_calc_mode == "median":
        best_number_of_epochs = round(cv_dic[cv_dic.index.str.startswith(f"trial_{best_trial_num_by_median}_")]["best_epoch"].mean())
        print("Best Trial by median is:", best_trial_num_by_median, f"with {metric_name}: {best_metric_by_median}")
        print("Best number of epochs by median is:", best_number_of_epochs)
        return best_number_of_epochs
        
def plot_history(history, name):
    fig, axs = plt.subplots(len(history.history.keys())//4, 2, figsize=(15, 20))

    for key, ax in zip([val for val in history.history.keys() if "val" not in val], axs.flat):
        for metric in [f"{key}", f"val_{key}"]:
            ax.plot(range(1, len(history.history[metric])+1), history.history[metric],"-o", label=metric)
            ax.legend()
            ax.set_xlabel("Epochs")
            
    fig.savefig(f"{name}_train_val_history.png")

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

def save_metrics_and_ROC_PR(model_name, model, training_x, training_y, testing_x, testing_y): 
    """Evaluate the model on the training_x and testing_x, then saves the results as pickled pandas name.pkl

    Args:
        model_name (string): Name of the pickled file to be saved.
        model (keras.model): A trained model.
        training_x (numpy.array): Training data (used to train the model).
        training_y (numpy.array): Target y's for the training data.
        testing_x (numpy.array): Training data that is already normalized and has appropriate dimentions.
        testing_y (numpy.array): Target y's for the testing data.
    """
    
    all_data_dic = dict()
    
    for x, y, name in zip([training_x, testing_x], [training_y, testing_y], ["training", "testing"]):
        
        #Get the model's metrics
        res = model.evaluate(x, y)
        res_dict = {metric:value for metric, value in zip(model.metrics_names, res)}
        
        #Get the Precision, Recall, FPR
        y_pred = model.predict(x)
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
        
        res_dict["precision_curve"] = precision.numpy()
        res_dict["fp_rate"] = fp_rate.numpy()
        res_dict["recall_curve"] = recall.numpy()
        res_dict["thresholds"] = m.thresholds
        
        all_data_dic[name] = res_dict
    
    pd.DataFrame.from_dict(all_data_dic, orient="index").to_pickle(f"keras_tuner_results/{model_name}.pkl")
    
def record_training_testing_indeces(model_name, training_val_indeces, testing_indeces):
    ids = training_val_indeces+testing_indeces
    labels = ["training_val"]*len(training_val_indeces) + ["testing"]*len(testing_indeces)
    pd.DataFrame(data = {"uniqids":ids, "category":labels}).sort_values(by=["category", "uniqids"]).reset_index(drop=True).to_csv(f"./keras_tuner_results/{model_name}_testing_training_indeces.csv")
    
