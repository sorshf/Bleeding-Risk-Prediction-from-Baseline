from data_preparation import prepare_patient_dataset
from constants import data_dir, instruction_dir, discrepency_dir, timeseries_padding_value, picled_objects, all_data_pics_path
import tensorflow.keras as keras
import tensorflow as tf
tf.random.set_seed(timeseries_padding_value)
import pandas as pd
from cross_validation import divide_into_stratified_fractions, get_X_y_from_indeces, normalize_training_validation, kfold_cv
import collections
from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif, mutual_info_classif
import pickle
import regex as re
import os


def get_important_feature_dic_baseline(baseline_dataframe):
    #This function is for feature selection
    def get_important_features(X, y, method, number_features):
        transformer = GenericUnivariateSelect(method, mode='k_best', param=number_features)
        X_new = transformer.fit(X, y)
        return X_new.get_feature_names_out()
    
    features_dict = dict()
    methods_list = [chi2, f_classif, mutual_info_classif]
    numbers_list = [1, 5, 10, 15, 20, 30, 50, 80]

    for method in methods_list:
        for number in numbers_list:
            features_dict[f"{number}_{method.__name__}"] = get_important_features(baseline_dataframe, 
                                                                        target_list, 
                                                                        method, 
                                                                        number)

    features_dict["total"] = baseline_dataframe.columns
    
    return features_dict

def prepare_training_validation_testing_data(patient_dataset):
    #Get the list of FUPS, Baseline, and Target
    FUPS_dict, FUPS_columns, Baseline_list, Baseline_list_columns, target_list = patient_dataset.get_data_x_y()
    
    print(len(FUPS_dict), len(FUPS_columns), len(Baseline_list), len(Baseline_list_columns), len(target_list))
    
    
    baseline_dataframe = pd.DataFrame(Baseline_list, columns=Baseline_list_columns)
    baseline_dataframe.index = baseline_dataframe["uniqid"].astype(int)
    baseline_dataframe = baseline_dataframe.drop(columns=["uniqid", "dtbas", "vteindxdt", "stdyoatdt"], axis=1)
    target_series = pd.Series(target_list, index=baseline_dataframe.index.astype(int))

    #Divide all the data into training and testing portions (two parts)
    training_val_indeces, testing_indeces = divide_into_stratified_fractions(target_series, fraction=0.2)

    #Get the training-validation data
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series, 
                                                                            )

    fold_n = 1
    all_training_indeces = dict()
    all_validation_indeces = dict()
    
    train_val_data, test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                          validation_indeces = testing_indeces, 
                                                          baseline_data = baseline_dataframe, 
                                                          FUPS_data_dic = FUPS_dict, 
                                                          all_targets_data = target_series, 
                                                          timeseries_padding_value=timeseries_padding_value)
    train_val_baseline_X, train_val_fups_X, train_val_y = train_val_data
    test_baseline_X, test_fups_X, test_y = test_data
    
    for training_data, validation_data in kfold_cv(baseline_train_val_X, fups_train_val_X.copy(), train_val_y):
        baseline_train_X, fups_train_X, train_y = training_data
        baseline_valid_X, fups_valid_X, valid_y = validation_data
        
        print(fups_train_X[0][0][0:5])
        print(f"The training set contains {len(baseline_train_X)} samples. {collections.Counter(train_y)}")
        #print(f"The testing set contains {len(baseline_test_X)} samples. {collections.Counter(test_y)}")
        print(f"The validation set contains {len(baseline_valid_X)} samples. {collections.Counter(valid_y)}")
        #print("testing_indeces",list(test_y[0:9].index))
        print("training_indeces",list(train_y[0:9].index))
        print("validation_indeces",list(valid_y[0:9].index))
        print("###############")
        all_training_indeces[fold_n] = list(train_y.index)
        all_validation_indeces[fold_n] = list(valid_y.index)
        fold_n += 1
        
    for fold in [1,2,3,4,5]:
        training_indeces_fold = set(all_training_indeces[fold])
        validation_indeces_fold = set(all_validation_indeces[fold])
        print(len(training_indeces_fold), len(validation_indeces_fold))
        print(f"Fold: {fold} -> Total Numbers {len(training_indeces_fold) + len(validation_indeces_fold)} -> Common ids in train-val {len(training_indeces_fold.intersection(validation_indeces_fold))}")
        print("###")
    
    return (train_val_baseline_X, train_val_fups_X, train_val_y), (test_baseline_X, test_fups_X, test_y)

def pickle_objects(mode, object, object_name):
    if mode=="w":
        with open(f"{picled_objects}_{object_name}.pkl", "wb") as file:
            pickle.dump(object, file) 
    elif mode == "r":
        with open(f"{picled_objects}_{object_name}.pkl", "rb") as file:
            object = pickle.load(file) 
        return object

def generate_pics_for_all(patient_dataset):
    ids_with_pics = [re.findall('[0-9]+', file) for file in os.listdir(all_data_pics_path)]
    ids_with_pics = [int(file[0]) for file in ids_with_pics if len(file)!=0]
    for patient in patient_dataset.all_patients:
        if patient.uniqid not in ids_with_pics:
            #print("+++",patient.uniqid)
            patient.plot_all_data(path= all_data_pics_path, 
                                    instruction_dir=instruction_dir, 
                                    patient_dataset= patient_dataset)
            
            
            print(f"{patient.uniqid} picture is done!")


def main():
    
    dataset_mode = "w"
    
    if dataset_mode == "w":
        patient_dataset = prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir)
        dic_of_removed_patients = patient_dataset.filter_patients_sequentially()
        patient_dataset.sort_patients()
        patient_dataset.correct_FUPPREDICTION_with_new_columns()

        #print(len(patient_dataset.all_patients))
        
        #Save the patient_dataset object
        pickle_objects(mode="w", object=patient_dataset, object_name="patient_dataset")
        
    elif dataset_mode == "r":
        #Load the patient_dataset
        patient_dataset = pickle_objects(mode="r", object=None, object_name="patient_dataset")

    #(train_val_baseline_X, train_val_fups_X, train_val_y), (test_baseline_X, test_fups_X, test_y) = prepare_training_validation_testing_data(patient_dataset)

    #FUP_stats = patient_dataset.create_FUP_stats(mode="return")    

    
    
    print(len(patient_dataset.all_patients))
    generate_pics_for_all(patient_dataset)
    
    

    
if __name__=="__main__":
    main()
    
    