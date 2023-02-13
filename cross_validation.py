import collections
import math
import random
import copy
import numpy as np
import tensorflow as tf

def divide_into_stratified_fractions(y, fraction):
    """Divide a binary series consists of 0 and 1 into two stratified sections according to the fractions, and return only their indeces.

    Args:
        y (Pandas.Series): A series object where the indeces are the patient ids, and values are their labels. (Must be bianry 0, 1)
        fraction (flaot): What fravtion of the initial values to be placed in the second list.

    Returns:
        list, list: The training indeces, the testing indeces.
    """
    index_1 = list(y[y==1].index)
    index_0 = list(y[y==0].index)
    
    random.seed(1375)
    np.random.seed(1375)

    random.shuffle(index_0)
    random.shuffle(index_1)

    number_of_training_1 = int(len(index_1) * (1-fraction))
    number_of_training_0 = int(len(index_0) * (1-fraction))

    training_indeces = index_1[:number_of_training_1] + index_0[:number_of_training_0]
    testing_indeces = index_1[number_of_training_1:] + index_0[number_of_training_0:]
    
    random.shuffle(training_indeces)
    random.shuffle(testing_indeces)
    
    return training_indeces, testing_indeces

def kfold_indeces(y, k):
    """Yields k-fold stratified (training, testing) indeces for a binary series consists of 0 and 1.

    Args:
        y (Pandas.Series): A series object where the indeces are the patient ids, and values are their labels. (Must be bianry 0, 1)
        k (int): Number of folds in k-fold CV.

    Yields:
        list, list: The training indeces, the testing indeces.
    """
 
    index_1 = list(y[y==1].index)
    index_0 = list(y[y==0].index)

    random.seed(1375)
    np.random.seed(1375)
    random.shuffle(index_0)
    random.shuffle(index_1)
    
    for ones, zeros in zip(np.array_split(index_1, k), np.array_split(index_0, k)):
        testing = set(ones).union(set(zeros))
        testing = list(testing)
        random.shuffle(testing)
        
        training = (set(index_1) - set(ones)).union((set(index_0) - set(zeros)))
        training = list(training)
        random.shuffle(training)
        
        
        yield training, testing

def normalize_training_validation(training_indeces, 
                                  validation_indeces, 
                                  baseline_data, 
                                  FUPS_data_dic, 
                                  all_targets_data, 
                                  timeseries_padding_value = -5.):
    """Returns the training and validation data (after normalization) for the baseline and follow-ups data. 
       The training data and validation data can also be the training_val data and testing data.
       Note that the training data (or training-val) here is used to calculate the mean and std for normalization of the validation (or testing) data.

    Args:
        training_indeces (list): List of indeces of the panda dataframe (which are actually patient IDs) for the training set.
        validation_indeces (list): List of indeces of the panda dataframe (which are actually patient IDs) for the validation set.
        baseline_data (pandas.DataFrame): All the baseline data.
        FUPS_data_dic (dict): The dictionary of all the FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        all_targets_data (pandas.Series): The labels of all the data.
        timeseries_padding_value (float, optional): The value used to pad the timeseries data. Defaults to -5.0 .

    Yields:
        _type_: (baseline_train_X, fups_train_X, train_y), (baseline_valid_X, fups_valid_X, valid_y)
    """
    
    baseline_data = baseline_data.copy()
    FUPS_data_dic = copy.deepcopy(FUPS_data_dic)
    
    #Get the training data
    baseline_train_X, fups_train_X, train_y = get_X_y_from_indeces(indeces = training_indeces, 
                                                                    baseline_data = baseline_data, 
                                                                    FUPS_data_dic = FUPS_data_dic, 
                                                                    all_targets_data = all_targets_data)
    #Normalize the baseline training data
    baseline_train_X, mean_baseline_training, std_baseline_training = normalize_data(baseline_train_X, 
                                                                                    is_timeseries=False,
                                                                                    is_training=True,
                                                                                    mean=None,
                                                                                    std=None,
                                                                                    timeseries_padding_value=None)
    
    #Normalize the timeseries FUP training data
    fups_train_X, mean_FUP_training, std_FUP_training = normalize_data(list(fups_train_X.values()), 
                                                                        is_timeseries=True,
                                                                        is_training=True,
                                                                        mean=None,
                                                                        std=None,
                                                                        timeseries_padding_value=timeseries_padding_value)
    

    ####################################################################################
    
    #Get the validation data
    baseline_valid_X, fups_valid_X, valid_y = get_X_y_from_indeces(indeces = validation_indeces, 
                                                                    baseline_data = baseline_data, 
                                                                    FUPS_data_dic = FUPS_data_dic, 
                                                                    all_targets_data = all_targets_data)
    
    #Normalize the baseline validation data
    baseline_valid_X = normalize_data(baseline_valid_X, 
                                    is_timeseries=False,
                                    is_training=False,
                                    mean = mean_baseline_training,
                                    std = std_baseline_training,
                                    timeseries_padding_value=None)
    
    #Normalize the timeseries FUP validation data
    fups_valid_X = normalize_data(list(fups_valid_X.values()), 
                                is_timeseries=True,
                                is_training=False,
                                mean=mean_FUP_training,
                                std=std_FUP_training,
                                timeseries_padding_value=timeseries_padding_value)
    
    
    return (baseline_train_X, fups_train_X, train_y), (baseline_valid_X, fups_valid_X, valid_y)

def kfold_cv(training_val_x_baseline, 
             training_val_x_FUPS, 
             training_val_y,
             k =5, 
             timeseries_padding_value =-5.) :
    """Yields k-fold training and validation portions (which are appropriately normalized) for training and hyperparameter optimization.

    Args:
        training_val_x_baseline (pandas.DataFrame): The training dataframe.
        training_val_x_FUPS (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        training_val_y (pandas.Series): The labels of the data.
        k (int, optional): Number of k-fold cv. Defaults to 5.
        timeseries_padding_value (float, optional): The value used to pad the timeseries data. Defaults to -5.0 .

    Yields:
        _type_: (baseline_train_X, fups_train_X, train_y), (baseline_valid_X, fups_valid_X, valid_y)
    """
    
    training_val_x_baseline_copy = training_val_x_baseline.copy()
    training_val_x_FUPS_copy = copy.deepcopy(training_val_x_FUPS)
    
    
    #Perform k-fold cross-validation
    for fold_training_idx, fold_validation_idx in kfold_indeces(training_val_y, k):
        
        (baseline_train_X, fups_train_X, train_y), (baseline_valid_X, fups_valid_X, valid_y) = normalize_training_validation(training_indeces = fold_training_idx, 
                                                                                                                            validation_indeces = fold_validation_idx, 
                                                                                                                            baseline_data = training_val_x_baseline_copy, 
                                                                                                                            FUPS_data_dic = training_val_x_FUPS_copy, 
                                                                                                                            all_targets_data = training_val_y, 
                                                                                                                            timeseries_padding_value=-5.)
        
        
        yield  (baseline_train_X, fups_train_X, train_y), (baseline_valid_X, fups_valid_X, valid_y)
        
def get_timeseries_mean_std(X_train):
    """Calculates the mean and std of a 3 Dimentional timeseries data of shape (sample, time, features).

    Args:
        X_train (list): list of lists containing the timeseries data.

    Returns:
        np.array, np.array: Two arrays of mean and std.
    """
    #decompose the list of lists into one list.
    X_train_copy = [item for patient in X_train.copy() for item in patient]
    
    #Reshape the training data from 3D to 2D
    X_train_copy = np.array(X_train_copy)
    
    #Calculate the mean and std of the data
    mean = np.mean(X_train_copy, axis=0, dtype=np.float64)
    std = np.std(X_train_copy, axis=0, dtype=np.float64)
    
    #Any standard deviation that was equal to 0, will become 1 to avoid divide by zero error when normalizing.
    for i, value in enumerate(std):
        if value == 0:
            std[i,] = 1 
    
    return mean, std

def get_timeseries_max_min(X_train):
    """Calculates the max and min of a 3 Dimentional timeseries data of shape (sample, time, features).

    Args:
        X_train (list): list of lists containing the timeseries data.

    Returns:
        np.array, np.array: Two arrays of max and min.
    """
    #decompose the list of lists into one list.
    X_train_copy = [item for patient in X_train.copy() for item in patient]
    
    #Reshape the training data from 3D to 2D
    X_train_copy = np.array(X_train_copy)
    
    #Calculate the mean and std of the data
    max = np.max(X_train_copy, axis=0)
    min = np.min(X_train_copy, axis=0)
    
    print(min)
    print(max)
    
    
    return max, min


def normalize_data(X_train, 
                   is_timeseries, 
                   is_training, 
                   mean,
                   std,
                   timeseries_padding_value):
    """Normalize the data for Timeseries data (array of shape (sample, time, features)), or non timeseries ones
       (array of shape (sample, features)).

    Args:
        X_train (np.array): Numpy array of data.
        is_timeseries (bool): Whether this is a timesereis data.
        is_training (bool): Whether this is the training set (so we can get its mean and std)
        mean (np.array): Mean of the features (this is only used when is_training==False)
        std (np.array): STD of the features (this is only used when is_training==False)

    Returns:
        np.array: Normalized array of the data
    """
    
    if is_timeseries:
        
        #Copy the data
        X_train_copy = X_train.copy()
        
        #For the training, we get its mean and std, otherwise mean and std should be provided
        if is_training:
            mean, std = get_timeseries_mean_std(X_train_copy)  #Get the mean and std
            #max, min = get_timeseries_max_min(X_train_copy)  #Get the max and min of data
            #Pad the timeseries data with the padding value
            X_train_copy = tf.keras.preprocessing.sequence.pad_sequences(X_train_copy, padding="post", dtype='float32', 
                                                            value=timeseries_padding_value)
            
            #Normalize the data (We shouldn't touch the padding values, hence the if statement)
            for sample in range(len(X_train_copy)):
                for timeline in range(X_train_copy.shape[1]):
                    if (X_train_copy[sample][timeline]!=np.array([timeseries_padding_value]*X_train_copy.shape[2], dtype='float32')).all():
                        X_train_copy[sample][timeline] -=  mean
                        X_train_copy[sample][timeline] /=  std
                        
                        #X_train_copy[sample][timeline] = (X_train_copy[sample][timeline] - min) / (max - min)
                        
                        
        #If it is not training, we use the mean and std to normalize the data, and then pad the sequence
        else:
            new_patient_list = []
            for patient in X_train_copy:
                new_timeline_list = []
                
                for time in patient:
                    time -=  mean
                    time /=  std                    
                    new_timeline_list.append(time)
                new_patient_list.append(new_timeline_list)
                
            X_train_copy = new_patient_list
            X_train_copy = tf.keras.preprocessing.sequence.pad_sequences(X_train_copy, dtype="float32", padding="post", value=timeseries_padding_value)
            
            
        

        
        #If it is training, also return the mean and std to be used for validation and testing
        #Also return max and min
        if is_training:
            return X_train_copy, mean, std
        else:
            return X_train_copy
    
    
    if not is_timeseries:
        
        #Make a copy of the data
        X_train_copy = X_train.copy()
        
        #For the training, we get its mean and std, otherwise mean and std should be provided
        if is_training:
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
        
        X_train_copy -=  mean
        X_train_copy /=  std
        
        #If it is training, also return the mean and std to be used for validation and testing
        if is_training:
            return X_train_copy, mean, std
        else:
            return X_train_copy
    
def get_X_y_from_indeces(indeces, 
                         baseline_data, 
                         FUPS_data_dic, 
                         all_targets_data):
    """Retrieve the baseline_x, fups_x, and the labels from the indeces.

    Args:
        indeces (list): List of indeces of the panda dataframe (which are actually patient IDs)
        baseline_data (pandas.DataFrame): The Baseline dataset.
        FUPS_data_dic (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        all_targets_data (pandas.Series): Pandas Series of all the patients labels.

    Returns:
        pandas.DataFrame, dict, Pandas.Series: Baseline Dataframe, Dict of FUPS (timeseries), a series of labels
    """
    #Creating a copy of dic is necessary to prevent referencing the original dic for processing
    FUPS_data_dic_copy = copy.deepcopy(FUPS_data_dic)
    
    baseline_X = baseline_data.loc[indeces,:]
    fups_X = {uniqid:FUPS_data_dic_copy[uniqid] for uniqid in indeces}
    
    #We pad the timeseries data only if it is the training set
    # if is_training:
    #     fups_X = tf.keras.preprocessing.sequence.pad_sequences(fups_X, padding="post", dtype='float32', 
    #                                                         value=timeseries_padding_value)
    y = all_targets_data[indeces]
    
    return baseline_X, fups_X, y

