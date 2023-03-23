#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""This module contains all the constants such as path names used in the project."""
# =============================================================================
# Imports
data_dir = "./Raw Data/data transferred Soroush with pwd.xlsx"
instruction_dir = "./Raw Data/column_preprocessing_excel test_Mar13_2022.xlsx"
discrepency_dir = "./Chantal_bleeding_revisions/2022-10-03  discrepencies_bleeding_data_Updated.xlsx"


timeseries_padding_value = -5.

picled_objects = "./pickle_objects/"

all_data_pics_path = "./all_data_pics/"

#Directory to save the excel or csv files containing generated tables and stats 
stats_directory = "./Stats/"

#These dictionaries are for graphing the patients on timelines
COLOR_dic = {
    "majbldconf":"red",
    "relbldconf":"orange",
    "nobld":"khaki",
    "fups":"grey",
    "fups_disc":"black",
    "baseline":"green"
}

SYMBOL_dic = {
    "majbldconf":"\u2715",
    "relbldconf":"\u2715",
    "nobld":"\u2715",
    "fups":"⦁",
    "fups_disc":"\u2620",
    "baseline":"⦁"
}

SIZE_dic = {
    "majbldconf":80,
    "relbldconf":80,
    "nobld":80,
    "fups":50,
    "fups_disc":80,
    "baseline":50
}

ALPHA_dic = {
    "majbldconf":0.9,
    "relbldconf":0.9,
    "nobld":0.9,
    "fups":0.6,
    "fups_disc":0.9,
    "baseline":0.6
}

#Number of Iterations for hyperparameter tuning and cv
NUMBER_OF_ITERATIONS_CV = 5