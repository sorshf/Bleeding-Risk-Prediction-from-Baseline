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