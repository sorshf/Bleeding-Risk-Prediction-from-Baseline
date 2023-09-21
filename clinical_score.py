#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""The module to produce predictions by common Clinical Prediction Score.

Usage:
    python clinical_score.py

"""
# =============================================================================
# Imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from data_preparation import prepare_patient_dataset
from constants import data_dir, instruction_dir, discrepency_dir
import os
from hypermodel_experiments import divide_into_stratified_fractions, get_X_y_from_indeces
from hypermodel_experiments import record_training_testing_indeces
from data_preparation import get_abb_to_long_dic


class ClinicalScore(BaseEstimator, ClassifierMixin):

    def __init__(self, model_name):
        self.model_name = model_name

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self
    
    def get_CHAP_Score(self, X):
        creatinine = X['Creatinine (umol/L)'] * 0.0010
        hemoglobin = X['Hemoglobin (g/L)'] * (-0.0122)
        age = X['Age Baseline'] * 0.0256
        antiplatelate = (1 if X['Antiplatelet Agent']==1 else 0) * 0.8786
        
        return 0.0195*np.exp(creatinine+hemoglobin+age+antiplatelate)
    
    def get_ACCP_Score(self, X):
        age = 1 if 75>X['Age Baseline']>65 else (2 if X['Age Baseline']>=75 else 0)
        
        previous_gas_bleed = 1 if X['Prior Gastrointestinal Bleed']==1 else 0
        
        renal_failure = 1 if X['Creatinine (umol/L)']>106.1 else 0
        
        previous_stroke = 1 if X['Prior Stroke (CVA)']==1 else 0
        
        diabete = 1 if X['Diabetes Mellitus']==1 else 0
        
        if ((X['Gender'] == 0) & (X['Hemoglobin (g/L)']<130)) | ((X['Gender'] == 1) & (X['Hemoglobin (g/L)']<120)):
            anemia = 1
        else:
            anemia = 0
            
        antiplatelate = 1 if X['Antiplatelet Agent']==1 else 0
        
        nseid = 1 if X['NSAIDs']==1 else 0
        
        TTR = 1 if X['INR Result'] > 3 else 0 #Note this is not in the actual calculator
        
        return age+previous_gas_bleed+renal_failure+previous_stroke+diabete+anemia+antiplatelate+nseid+TTR
    
    def get_RIETE_Score(self, X):
        
        age = 1 if X['Age Baseline']>75 else 0
        
        renal_failure = 1.5 if X['Creatinine (umol/L)']>106.1 else 0
        
        if ((X['Gender'] == 0) & (X['Hemoglobin (g/L)']<130)) | ((X['Gender'] == 1) & (X['Hemoglobin (g/L)']<120)):
            anemia = 1.5
        else:
            anemia = 0
        
        
        if (X['PE only']==1) | (X['DVT and PE']==1):
            Symptomatic_pulmonary_embolism = 1
        else:
            Symptomatic_pulmonary_embolism = 0
            
        return age+renal_failure+anemia+Symptomatic_pulmonary_embolism
            
    def get_VTE_BLEED_Score(self, X):
        
        age = 1.5 if X['Age Baseline']>=60 else 0
        
        previous_gas_bleed = 1.5 if X['Prior Gastrointestinal Bleed']==1 else 0
        
        renal_failure = 1.5 if X['Creatinine (umol/L)']>106.1 else 0
        
        if ((X['Gender'] == 0) & (X['Hemoglobin (g/L)']<130)) | ((X['Gender'] == 1) & (X['Hemoglobin (g/L)']<120)):
            anemia = 1.5
        else:
            anemia = 0
        
        hypertension_med = 1 if ((X['Current Treatment for Hypertension']==1) & (X['Gender'] == 0)) else 0
        
        return age+previous_gas_bleed+renal_failure+anemia+hypertension_med
    
    
    def get_HAS_BLED_Score(self, X):
        age = 1 if X['Age Baseline']>65 else 0
        
        previous_gas_bleed = 1 if X['Prior Gastrointestinal Bleed']==1 else 0
        
        renal_failure = 1 if X['Creatinine (umol/L)']>200 else 0
        
        previous_stroke = 1 if X['Prior Stroke (CVA)']==1 else 0
        
        antiplatelate = 1 if X['Antiplatelet Agent']==1 else 0
        
        TTR = 1 if X['INR Result'] > 3 else 0 #Note this is not in the actual calculator

        hypertension_med = 1 if X['Current Treatment for Hypertension']==1 else 0
        
        return age+previous_gas_bleed+renal_failure+previous_stroke+antiplatelate+TTR+hypertension_med
    
    
    def get_OBRI_Score(self, X):
        age = 1 if X['Age Baseline']>=65 else 0 #On OBRI paper, it is >= 65
        previous_stroke = 1 if X['Prior Stroke (CVA)']==1 else 0
        previous_gas_bleed = 1 if X['Prior Gastrointestinal Bleed']==1 else 0
        
        renal_failure = 1 if X['Creatinine (umol/L)']>133 else 0
        
        diabete = 1 if X['Diabetes Mellitus']==1 else 0
        
        if ((X['Gender'] == 0) & (X['Hemoglobin (g/L)']<130)) | ((X['Gender'] == 1) & (X['Hemoglobin (g/L)']<120)):
            anemia = 1
        else:
            anemia = 0
        
        myocardial_infarction = 1 if X['Prior Myocardial Infarction']==1 else 0
        
        comorbidity = 1 if sum([renal_failure,diabete,anemia,myocardial_infarction])>0 else 0
        
        return age+previous_gas_bleed+previous_stroke+comorbidity
        
        
    def predict(self, X, mode="decision"):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        #X = check_array(X)
        
        if self.model_name == 'CHAP':
            predictions_score = X.apply(lambda val: self.get_CHAP_Score(val), axis=1)
            predictions_decision = np.where(predictions_score >= 0.025, 1, 0)
        elif self.model_name == 'ACCP':
            predictions_score = X.apply(lambda val: self.get_ACCP_Score(val), axis=1)
            predictions_decision = np.where(predictions_score >= 2, 1, 0)
        elif self.model_name == 'RIETE':
            predictions_score = X.apply(lambda val: self.get_RIETE_Score(val), axis=1)
            predictions_decision = np.where(predictions_score >= 5, 1, 0)
        elif self.model_name == 'VTE-BLEED':
            predictions_score = X.apply(lambda val: self.get_VTE_BLEED_Score(val), axis=1)
            predictions_decision = np.where(predictions_score >= 2, 1, 0)
        elif self.model_name == 'HAS-BLED':
            predictions_score = X.apply(lambda val: self.get_HAS_BLED_Score(val), axis=1)
            predictions_decision = np.where(predictions_score >= 3, 1, 0)
        elif self.model_name == 'OBRI':
            predictions_score = X.apply(lambda val: self.get_OBRI_Score(val), axis=1)
            predictions_decision = np.where(predictions_score >= 3, 1, 0)


        if mode == 'decision':
            return predictions_decision
        elif mode == 'score':
            return predictions_score
        
    def predict_proba(self, X):
        if self.model_name == 'CHAP':
            predictions_score = X.apply(lambda val: self.get_CHAP_Score(val), axis=1)
            
        return predictions_score
        

def main():
    #Read the patient dataset
    patient_dataset = prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir)

    #Remove patients without baseline, remove the FUPS after bleeding/termination, fill FUP data for those without FUP data
    patient_dataset.filter_patients_sequentially(mode="fill_patients_without_FUP")
    print(patient_dataset.get_all_targets_counts(), "\n")

    #Add one feature to each patient indicating year since baseline to each FUP
    patient_dataset.add_FUP_since_baseline()

    #Get the BASELINE, and Follow-up data from patient dataset
    FUPS_dict, _, baseline_dataframe, target_series = patient_dataset.get_data_x_y(baseline_filter=["uniqid", "dtbas", "vteindxdt", "stdyoatdt"], 
                                                                                                FUP_filter=[])

    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    for model_name in ['CHAP','ACCP','RIETE','VTE-BLEED','HAS-BLED','OBRI']:
        #Create a dir to save the result of experiment
        if not os.path.exists(f"./keras_tuner_results/{model_name}"):
            os.makedirs(f"./keras_tuner_results/{model_name}")

        #Record the training_val and testing indeces
        record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)

        #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
        #train_val data are used for hyperparameter optimization and training.
        baseline_test_X, _, test_y = get_X_y_from_indeces(indeces = testing_indeces, 
                                                                                baseline_data = baseline_dataframe, 
                                                                                FUPS_data_dic = FUPS_dict, 
                                                                                all_targets_data = target_series)
        
        #Change the names of the columns in the baseline data to make them compatible with the clinical score
        abb_to_long_dic = get_abb_to_long_dic(instructions_dir=instruction_dir, CRF_name="BASELINE")
        baseline_test_X.columns = [abb_to_long_dic[col] if col in abb_to_long_dic else col for col in baseline_test_X.columns]

        #Create the score object
        mychap_clf = ClinicalScore(model_name)

        #Fitting doesn't do anything special, it is just rerequired for the class
        mychap_clf = mychap_clf.fit(baseline_test_X, test_y)

        #Record exactly what are the predictions for each sample on the test dataset
        y_pred_classes = mychap_clf.predict(baseline_test_X)
        y_pred = mychap_clf.predict(baseline_test_X, mode="score")
        
        number_of_FUP = [len(FUPS_dict[uniqid]) for uniqid in list(test_y.index)]
        record_dict = {"uniqid":list(test_y.index),"FUP_numbers":number_of_FUP, "y_actual":test_y.values, "y_pred":y_pred,
                    "y_pred_classes":y_pred_classes}

        #Save the detailed results
        pd.DataFrame(record_dict).to_csv(f"keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
        