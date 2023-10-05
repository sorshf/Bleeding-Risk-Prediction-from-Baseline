#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""The module contains the functions to read and prepare raw data."""
# =============================================================================
# Imports
import pandas as pd
import numpy as np
from data_classes import Dataset
import re
from constants import instruction_dir, data_dir, discrepency_dir

def get_type_dic(df_instructions):
    """Gets a pandas dataframe (with columns "Abbreviations" and "Type")
        returns a dictionary with keys being abbreviations name and values being appropriate
        Python type.

    Returns:
        dict: A dictionary with the abbreviation:python_type structure.
    """
    
    type_dic = dict()
    for row in df_instructions.iterrows():
        col_name = row[1]["Abbreviation"]
        type_ = row[1]["Type"]
        if type_ == "int":
            type_dic[col_name] = "Int64"
        elif type_ == "float":
            type_dic[col_name] = float
        elif type_ == "object":
            type_dic[col_name] = object
        elif type_ == "Date":
            type_dic[col_name] = 'datetime64[D]'
            
    return type_dic

def fill_na_in_df(df_instructions, CRF_name, df_data):
    """Function that gets instruction pandas dataframe (must contain columns "Abbreviations", "Type", 
        "Fill-na", "Need-1", "CRF_name"), the name of the CRF, and the dataset.
        Returns the dataset with the NAs filled according to the instructions.
    
    Returns:
        pandas.df: The pandas dataframe of df_data with NAs filled.
    """
    
    #Fill NaNs in the columns that are 'Mode', 'Zero', or 'Median'
    #Get the columns names that need 'Mode' and 'Zero' and 'Median' Fill-na
    cols_fillna_MoZeMe = df_instructions[(df_instructions['Fill-na']=='Mode')|
                                      (df_instructions['Fill-na']=='Zero')|
                                      (df_instructions['Fill-na']=='Median')&
                                      (df_instructions["Need-1"]=="Yes")&
                                     (df_instructions["CRF_name"]==CRF_name)]["Abbreviation"]      
 
    for col_name in cols_fillna_MoZeMe:
        fill_na_type = str(df_instructions[df_instructions['Abbreviation']==col_name]['Fill-na'].values[0])
        if fill_na_type == "Mode":
            primary_type = df_data[col_name].dtype
            df_data[col_name] = df_data[col_name].apply(lambda x: list(df_data[col_name].mode())[0] if pd.isna(x) else x)
            if primary_type != object:
                df_data[col_name] = df_data[col_name].astype(primary_type)

        elif fill_na_type == "Zero":
            df_data[col_name] = df_data[col_name].apply(lambda x: 0 if pd.isna(x) else x)
            df_data[col_name] = df_data[col_name].astype(int)

        elif fill_na_type == "Median":
            df_data[col_name] = df_data[col_name].apply(lambda x: df_data[col_name].median() if pd.isna(x) else x)
            
    return df_data

def get_encoding_dic(encoding):
    """Converts the encoding string in the instructions excel file into a dic of int:string.

    Args:
        encoding (string): Encoding string in the instructions excel.

    Returns:
        dic: Dict of int:string generated from encoding dict.
    """
    #Keys are the numbers (0,1,2,..) and values are corresponding values (No, Yes Unknown, ...)
    encoding_dic = dict()
    for ele in encoding.split(','):
        first_ele = ele.split(":")[0].replace('(','').replace('[','').replace(' ','').replace('"', "").replace("'", "").replace('‘',"").replace('’',"")
        second_ele = ele.split(":")[1].replace(')','').replace(']','').replace(' ','').replace('"', "").replace("'", "").replace('‘',"").replace('’',"")
        encoding_dic[first_ele] = second_ele
    return encoding_dic

def one_hot_encode(df_instructions, df_data):
    """One-hot encode a dataframe as instructed in the instructions excel.

    Args:
        df_instructions (pandas.df): The instructions excel as dataframe.
        df_data (pandas.df): The data as dataframe.

    Returns:
        pandas.df: Data with the one-hot encoded features.
    """
    #Encode the columns that will be One-Hot encoded
    cols_one_hot_name = df_instructions[df_instructions["Preprocessing"]=="One-hot"]["Abbreviation"]

    #For the columns that need to be one-hot-encoded, convert the integers to string encoded values
    for col_name in cols_one_hot_name:
        if col_name in df_data.columns:
            encoding = str(df_instructions[df_instructions["Abbreviation"]==col_name]["Encoding"].values)
            encoding_dic = get_encoding_dic(encoding)

            #Use the encoding dic to convert the integers to strings
            for key in encoding_dic:
                df_data[col_name] = df_data[col_name].astype(object)
                df_data[col_name] = df_data[col_name].replace(to_replace={int(key):encoding_dic[key]})
                
    #One-hot encode the columns that need to be one-hot encoded
    for col in cols_one_hot_name:
        if col in df_data.columns:
            one_hotted = pd.get_dummies(df_data[col], prefix = col)
            df_data = pd.concat([df_data, one_hotted], axis=1)
            del(df_data[col])
            
    return df_data

def get_abb_to_long_dic(instructions_dir, CRF_name):
    """Create a dictionary of abbreviation:LongName from the instructions excel.

    Args:
        instructions_dir (string): Path to instruction dir excel.
        CRF_name (string): Name of the CRF we want the abb_to_long for.

    Returns:
        dict: abbreviation:LongName dictionary.
    """
    #Read the excel file for instructions
    instructions = pd.read_excel(instructions_dir)
    
    instructions = instructions[(instructions["CRF_name"]==CRF_name)]

    #The dictionary for each abbreviation to long name
    abb_to_longName_dic = dict()

    for col_name in instructions["Abbreviation"]:
        long_name = list(instructions[instructions["Abbreviation"]==col_name]['Long_name'])[0]
        abb_to_longName_dic[col_name] = long_name

    return abb_to_longName_dic

def remove_incorrect_values(dataframe, instructions, verbose=True):
    """Replace the wrong values with None as instructed in the instructions encoding.

    Args:
        dataframe (pandas.df): The data that is not yet one-hot encoded.
        instructions (pandas.df): The instructions excel file.
        verbose (boolean): Whether print the wrong values.

    Returns:
        pandas.df: The dataframe with the weird/wrong values converted to None.
    """
    #There are weird/wrong values in some columns of the the dataset. 
    #We use the encoding to only keep the relevant values, then make the rest of the values NA
    for col_name in dataframe.columns:
        if col_name in list(instructions["Abbreviation"]):
            encoding = str(instructions[instructions["Abbreviation"]==col_name]["Encoding"].values)
            if encoding != "['None']":
                encoding_dic = get_encoding_dic(encoding)
                valid_numbers = encoding_dic.keys()
                values_in_df = list(dataframe[col_name].unique())
                for value in values_in_df:
                    if (str(value) not in valid_numbers) & (~pd.isna(value)):
                        if verbose:
                            print(f"The value {value} should not be in column {col_name} in {list(instructions['CRF_name'].unique())[0]}. So it is replaced with NA.")
                        dataframe[col_name] = dataframe[col_name].replace(value, None)
    return dataframe

def prepare_FUPOUTCOME(data_dir, instruction_dir, discrepency_dir):
    """Read the FUPOUTCOME raw dataset, correct discrepencies, replace wrong values with None, 
    and then fill NAs as instructed in the instructions csv file.

    Args:
        data_dir (string): Path to the raw data excel file.
        instruction_dir (string): Path to the instructions excel file.
        discrepency_dir (string): Path to the discrepency excel file.

    Returns:
        pandas.df: FUPOUTCOME dataframe.
    """
    #Read the instruction csv of which features to use and how to process them
    instructions = pd.read_excel(instruction_dir)
    instructions = instructions[(instructions["CRF_name"]=="FUPOUTCOME")&(instructions["Need-1"]=="Yes")]

    #Create the dictionary of the types for each feature in the instructions csv file
    type_dic = get_type_dic(instructions)
    
    #Read the data, and then fix the discrepencies
    df_FUPOUTCOME = correct_FUPOUTCOME_discrepencies(data_dir, discrepency_dir)

    #Change the datatype of the columns according to the instructions
    df_FUPOUTCOME = df_FUPOUTCOME.astype(type_dic)
    
    #Only keep the values that we determined in the instructions
    df_FUPOUTCOME = df_FUPOUTCOME[type_dic.keys()]
      
    #There are weird/wrong values in some columns of the the dataset. 
    #We use the encoding to only keep the relevant values, then make the rest of the values NA
    df_FUPOUTCOME = remove_incorrect_values(df_FUPOUTCOME, instructions)

    #Fill_NAs
    df_FUPOUTCOME = fill_na_in_df(instructions, CRF_name="FUPOUTCOME", df_data=df_FUPOUTCOME)
    
    return df_FUPOUTCOME

def correct_FUPOUTCOME_discrepencies(data_dir, discrepency_dir):
    """Generates the dataframe for FUPOUTCOME with the data discrepencies fixed.

    Args:
        data_dir (string): The filepath  to the excel file with our data.
        discrepency_dir (string): The filepath to the discrepencies data by Chantal Rockwell.

    Returns:
        pd.dataframe: The dataframe for FUPOUTCOME.
    """
    #Read the Outcome dataset
    df_OUTCOME = pd.read_excel(data_dir, header=0, sheet_name = "FUPOUTCOME",  engine="openpyxl")

    #Remove all the rows with NAs in the dates in FUPOUTCOME
    df_OUTCOME.drop(df_OUTCOME[pd.isna(df_OUTCOME["fuodt"])].index, axis=0, inplace=True)

    #Get the correct data (or missing values) from the discrepency data for FUPOUTCOME
    discrepency_data_FUPOUTCOME = pd.read_excel(discrepency_dir, header=1, sheet_name = "FUPOUTCOME_missing",  engine="openpyxl")

    #Get the discrepency correction sheet (made by Soroush Fard)
    #Note: In this sheet I made decisions as to whether change the dates of the rows, or remove them.
    discrepency_data_correct_FUPS = pd.read_excel(discrepency_dir, header=0, sheet_name = "correct_fups_soroush",  engine="openpyxl")

    #The portion of discrepency correction sheet for FUPOUTCOME
    FUPS_to_correct_FUPOUTCOME = discrepency_data_correct_FUPS[discrepency_data_correct_FUPS["form"]=="FUPOUTCOME"]
    #Use the discrepency correction sheet, to remove unwanted entries, or change their dates accordingly
    for _, row in FUPS_to_correct_FUPOUTCOME.iterrows():
        if row["process"] == "remove":
            df_OUTCOME.drop(df_OUTCOME[(df_OUTCOME["fuodt"]==row["date"])&(df_OUTCOME["uniqid"]==row["uniqid"])].index, axis=0, inplace=True)
        else:
            df_OUTCOME.loc[(df_OUTCOME["fuodt"]==row["date"])&(df_OUTCOME["uniqid"]==row["uniqid"]), "fuodt"] = row["process"]
            
    #Get the missing data produced by Chantal Rockwell for the OUTCOME dataset
    discrepency_data_FUPOUTCOME = discrepency_data_FUPOUTCOME[discrepency_data_FUPOUTCOME["to-use"]=="Yes"]

    #Removing the useless data columns
    discrepency_data_FUPOUTCOME = discrepency_data_FUPOUTCOME.drop(labels=["to-use", "Notes"], axis=1)

    #Fill the nans with 0
    discrepency_data_FUPOUTCOME = discrepency_data_FUPOUTCOME.fillna(0)

    #Concatenate the additional data that was misssing to the original data
    df_OUTCOME = pd.concat([df_OUTCOME, discrepency_data_FUPOUTCOME])
    
    return df_OUTCOME

def prepare_FUPPREDICTOR(data_dir, instruction_dir, discrepency_dir):
    """Read the FUPPREDICTOR raw dataset, correct discrepencies, fix follow-up weights,
    replace wrong values with None, fill NAs as instructed in the instructions csv file,
    and then one-hot encode the variables as instructed.

    Args:
        data_dir (string): Path to the raw data excel file.
        instruction_dir (string): Path to the instructions excel file.
        discrepency_dir (string): Path to the discrepency excel file.

    Returns:
        pandas.df: FUPPREDICTOR dataframe.
    """
    #Read the instruction Excel of which features to use and how to process them
    instructions = pd.read_excel(instruction_dir)
    instructions = instructions[(instructions["CRF_name"]=="FUPPREDICTOR")&(instructions["Need-1"]=="Yes")]

    #Create the dictionary of the types for each feature in the instructions csv file
    type_dic = get_type_dic(instructions)

    #Read the data, and then fix the discrepencies
    df_FUPPREDICTOR = correct_FUPPREDICTOR_discrepencies(data_dir, discrepency_dir)

    #Change the datatype of the columns according to the instructions
    df_FUPPREDICTOR = df_FUPPREDICTOR.astype(type_dic)
    
    #Only keep the values that we determined in the instructions
    df_FUPPREDICTOR = df_FUPPREDICTOR[type_dic.keys()]

    #There are weird/wrong values in some columns of the the dataset. 
    #We use the encoding to only keep the relevant values, then make the rest of the values NA
    df_FUPPREDICTOR = remove_incorrect_values(df_FUPPREDICTOR, instructions)

    #Taking care of the NAs in the wtfukg (weights) in FUPPREDICTOR
    df_BASELINE_weights = pd.read_excel(data_dir, header=0, usecols=["wtbaskg", "uniqid"], sheet_name = "BASELINE",  engine="openpyxl")

    #For the follow-up weights, if it is missing, use the mean of the other follow-up weights
    #However, if all follow-up weights are missing, use the weights recorded in Baseline
    #If both follow-up weights and baseline weights are missing, use 75.0 as the weights in kg
    for ind in df_FUPPREDICTOR[df_FUPPREDICTOR["wtfukg"].isna()].index:
        patient_id = df_FUPPREDICTOR.loc[ind,"uniqid"] #Get the patient ID
        baseline_weight = df_BASELINE_weights[df_BASELINE_weights["uniqid"]==patient_id]["wtbaskg"].values[0]
        mean = np.mean([i for i in df_FUPPREDICTOR[df_FUPPREDICTOR["uniqid"]==patient_id]["wtfukg"] if ~pd.isna(i)])

        if pd.isna(mean) & ~pd.isna(baseline_weight):
            df_FUPPREDICTOR.loc[ind, "wtfukg"] = baseline_weight
        elif pd.isna(mean) & pd.isna(baseline_weight):
            df_FUPPREDICTOR.loc[ind, "wtfukg"] = 75.0
        else:
            df_FUPPREDICTOR.loc[ind, "wtfukg"] = mean

    #Fill the NAs in the dataset according to the instructions excel file
    df_FUPPREDICTOR = fill_na_in_df(instructions, CRF_name="FUPPREDICTOR", df_data=df_FUPPREDICTOR)
    
    #One-hot encode the columns that need to be one-hot encoded
    df_FUPPREDICTOR = one_hot_encode(instructions, df_FUPPREDICTOR)
        
    return df_FUPPREDICTOR

def correct_FUPPREDICTOR_discrepencies(data_dir, discrepency_dir):
    """Generates the dataframe for FUPPREDICTOR with the data discrepencies fixed.

    Args:
        data_dir (string): The filepath  to the excel file with our data.
        discrepency_dir (string): The filepath to the discrepencies data by Chantal Rockwell.

    Returns:
        pd.dataframe: The dataframe for FUPPREDICTOR.
    """
    #Read the FUPPREDICTOR data
    df_FUPPREDICTOR = pd.read_excel(data_dir, header=0, sheet_name = "FUPPREDICTOR",  engine="openpyxl")

    #Remove all the rows with NAs in the dates in FUPPREDICTOR
    df_FUPPREDICTOR.drop(df_FUPPREDICTOR[pd.isna(df_FUPPREDICTOR["fudt"])].index, axis=0, inplace=True)

    #Get rid of the duplicate values for 121134
    df_FUPPREDICTOR = df_FUPPREDICTOR.drop(df_FUPPREDICTOR[(df_FUPPREDICTOR["uniqid"]==121134) & (df_FUPPREDICTOR["fudt"]==pd.Timestamp("2014-05-27"))].index[0])

    #Remove the row with date 2001-07-04 for uniqid 101225
    #The column "fudt" has Timestamp('2001-07-04 00:00:00')
    df_FUPPREDICTOR.drop(df_FUPPREDICTOR[df_FUPPREDICTOR["fudt"]==pd.Timestamp('2001-07-04')].index, axis=0, inplace=True)

    #Get the correct data (or missing values) from the discrepency data for FUPPREDICTOR
    discrepency_data_FUPPREDICTOR = pd.read_excel(discrepency_dir, header=0, sheet_name = "FUPPREDICTOR_missing",  engine="openpyxl")

    #Get the discrepency correction sheet (made by Soroush Fard)
    #Note: In this sheet I made decisions as to whether change the dates of the rows, or remove them.
    discrepency_data_correct_FUPS = pd.read_excel(discrepency_dir, header=0, sheet_name = "correct_fups_soroush",  engine="openpyxl")

    #The portion of discrepency correction sheet for FUPPREDICTOR
    FUPS_to_correct_FUPPREDICTOR = discrepency_data_correct_FUPS[discrepency_data_correct_FUPS["form"]=="FUPPREDICTOR"]
    #Use the discrepency correction sheet, to remove unwanted entries, or change their dates accordingly
    for _, row in FUPS_to_correct_FUPPREDICTOR.iterrows():
        if row["process"] == "remove":
            df_FUPPREDICTOR.drop(df_FUPPREDICTOR[(df_FUPPREDICTOR["fudt"]==row["date"])&(df_FUPPREDICTOR["uniqid"]==row["uniqid"])].index, axis=0, inplace=True)
        else:
            df_FUPPREDICTOR.loc[(df_FUPPREDICTOR["fudt"]==row["date"])&(df_FUPPREDICTOR["uniqid"]==row["uniqid"]), "fudt"] = row["process"]

    #Get the missing data produced by Chantal Rockwell for the PREDICTOR dataset
    discrepency_data_FUPPREDICTOR = discrepency_data_FUPPREDICTOR[discrepency_data_FUPPREDICTOR["to-use"]=="Yes"]

    #Removing the useless data columns
    discrepency_data_FUPPREDICTOR = discrepency_data_FUPPREDICTOR.drop(labels=["to-use", "Notes"], axis=1)

    #Making sure we have the weights in kg
    discrepency_data_FUPPREDICTOR["wtfukg"] = discrepency_data_FUPPREDICTOR.apply(lambda x: round(x["wtfulbs"]/2.205, 1) if pd.isna(x["wtfukg"]) 
                                                                                else x["wtfukg"], axis=1)
    #Concatenate the additional data that was misssing to the original data
    df_FUPPREDICTOR = pd.concat([df_FUPPREDICTOR, discrepency_data_FUPPREDICTOR])
    
    return df_FUPPREDICTOR

def prepare_GENOTYPE(data_dir, instruction_dir):
    """Reads the Genotype raw dataset, fill NAs according to the instructions,
       and then one-hot encode the dataset.

    Args:
        data_dir (string): Path to the data excel file.
        instruction_dir (string): Path to the instructions directory.

    Returns:
        pandas.df: The genotype data.
    """
    #Read the instruction csv of which features to use and how to process them
    instructions = pd.read_excel(instruction_dir)
    instructions = instructions[(instructions["CRF_name"]=="GENOTYPE")&(instructions["Need-1"]=="Yes")]

    #Create the dictionary of the types for each feature in the instructions csv file
    type_dic = get_type_dic(instructions)

    #Read the Genotype dataset
    df_GENOTYPE = pd.read_excel(data_dir, header=0, usecols=type_dic.keys(), dtype=type_dic, sheet_name = "GENOTYPE",  engine="openpyxl")
    
    
    #Fill NAs according to the instructions
    df_GENOTYPE = fill_na_in_df(instructions, CRF_name="GENOTYPE", df_data=df_GENOTYPE)

    #One-hot encode the columns that need to be one-hot encoded
    for col in instructions[instructions["Preprocessing"]=="One-hot"]["Abbreviation"]:
        one_hotted = pd.get_dummies(df_GENOTYPE[col], prefix = col)
        df_GENOTYPE = pd.concat([df_GENOTYPE, one_hotted], axis=1)
        del(df_GENOTYPE[col])
        
    return df_GENOTYPE

def prepare_AD1(data_dir):
    """Read the AD1 data after doing some cleaning.

    Args:
        data_dir (string): Path to the data excel file.

    Returns:
        pandas.df: AD1 (target) data.
    """
    df_AD1 = pd.read_excel(data_dir, header=0, sheet_name = "AD1",  engine="openpyxl")

    #Remove rows in AD1 where both blddtadj and blddtadj_yy are NAs
    df_AD1 = df_AD1.drop(df_AD1[(df_AD1["blddtadj"].isna()) & (df_AD1["blddtadj_yy"].isna())].axes[0].values, axis=0)

    #There are missing dates "blddtadj" in AD1. So, we use the year as the date (January first of each year)
    df_AD1.loc[df_AD1["blddtadj"].isna(), "blddtadj"] = pd.to_datetime([str(int(i))+"-01-01" for i in df_AD1[df_AD1["blddtadj"].isna()]["blddtadj_yy"].values])
    
    return df_AD1

def prepare_FUPDISCONT(data_dir, instruction_dir):
    """Read the FUPDISCONT data.

    Args:
        data_dir (string): Path to the data excel file.
        instruction_dir (string): Path to the instructions excel file.

    Returns:
        pandas.df: FUPDISCONT data.
    """
    #Read the excel file for instructions to be performed on the columns of Baseline
    instructions = pd.read_excel(instruction_dir)
    instructions = instructions[(instructions["CRF_name"]=="FUPDISCONT")&(instructions["Need-1"]=="Yes")]
    
    #Create the dictionary of the types for each 
    type_dic = get_type_dic(instructions)
    
    #Read the data
    df_FUPDISCONT =  pd.read_excel(data_dir, usecols=list(type_dic.keys()), header=0, sheet_name = "FUPDISCONT",  engine="openpyxl", dtype=type_dic)
    
    return df_FUPDISCONT

def prepare_BASELINE(data_dir, instruction_dir, discrepency_dir):
    """Read the raw Baseline data, cleans the data, and then one-hot encode the necessary variables.

    Args:
        data_dir (string): Path to the data excel file.
        instruction_dir (string): Path to the instructions excel file. 
        discrepency_dir (string): Path to the discrepencies excel file.
    """
    #Read the excel file for instructions to be performed on the columns of Baseline
    instructions = pd.read_excel(instruction_dir)
    instructions = instructions[(instructions["CRF_name"]=="BASELINE")&(instructions["Need-1"]=="Yes")]

    #Create the dictionary of the types for each 
    type_dic = get_type_dic(instructions)

    #Read the data in Baseline
    df_BASELINE = pd.read_excel(data_dir,usecols=list(type_dic.keys()), header=0, sheet_name = "BASELINE",  engine="openpyxl", dtype=type_dic)
    
    #Read the Discrepency correction excel file
    discrepency_data_baseline = pd.read_excel(discrepency_dir, header=0, sheet_name = "Discrepencies in Baseline",  engine="openpyxl")
    
    #Using the discrepencies correction data, let's revise the numbers that need to be revised.
    for row in discrepency_data_baseline.iterrows():
        uniqid = row[1]["Uniqid"]
        feature = row[1]["Column (Feature) Name"]
        revised_value = row[1]["Revised Values"]
        
        #We are not interested to correct the values of the features that are not important (hence the if statement)
        if (feature in df_BASELINE.columns) & (~pd.isna(revised_value)):
            df_BASELINE.loc[df_BASELINE["uniqid"]==uniqid, feature] = revised_value
        

    #The patients with id 17XXXX are from US. There are few instances where their hemglbas are in g/dL instead of g/L. 
    #So, we fix them by multiplying g/dL*10.
    us_hemglbas_indeces = df_BASELINE.loc[(df_BASELINE["uniqid"].astype(str).str.startswith("17")) & (df_BASELINE["hemglbas"] < 20)].index
    df_BASELINE.loc[us_hemglbas_indeces, "hemglbas"] = df_BASELINE.loc[us_hemglbas_indeces, "hemglbas"].apply(lambda x: x*10)
    
    
    #The feature "ptsbas" is the same feature that was renamed later in 2010 to "currpts"(also its location was changed in the CRF form).
    #Furthermore, "ptsbas" has different encoding compared to "currpts"
    #[(‘1': ‘Yes’), (‘0’:’No’), (‘3’:’Unknown’)] compared to [(‘1': ‘Yes’), (‘0’:’No’), (‘2': ‘Unknown’)]
    df_BASELINE["currpts"] = df_BASELINE.apply(lambda x: x["currpts"] if pd.isna(x["ptsbas"]) else x["ptsbas"], axis=1)
    df_BASELINE["currpts"] = df_BASELINE["currpts"].fillna(df_BASELINE["currpts"].mode()[0]) #Fill the one NAs with the mode (zero)
    del(df_BASELINE["ptsbas"])


    #The feature "statcur" is the same feature that was renamed later in 2010 to "statin"
    df_BASELINE["statin"] = df_BASELINE.apply(lambda x: x["statin"] if pd.isna(x["statcur"]) else x["statcur"], axis=1)
    df_BASELINE["statin"] = df_BASELINE["statin"].fillna(df_BASELINE["statin"].mode()[0]) #Fill the one NAs with the mode (zero)
    del(df_BASELINE["statcur"])


    #Fillling NAs for diabmel and diabtyp 
    # diabmel	Diabetes Mellitus
    # diabtyp	Diabetes Type
    #There is one patient where there is no type for the diabetes (diabtyp is NA). Change its "diabmel" to 0
    df_BASELINE.loc[pd.isna(df_BASELINE["diabtyp"])&(df_BASELINE["diabmel"]==1), "diabmel"] = 0
    #Logic: All the people with positive diabmel, will either have Type1 or Type 2, otherwise diabmel, diabtyp - Type1, and diabtyp - Type2 are 0
    df_BASELINE['diabtyp_Type1'] = df_BASELINE.apply(lambda x: 1 if x['diabmel'] == 1 and x['diabtyp'] == 1 else 0, axis=1)
    df_BASELINE['diabtyp_Type2'] = df_BASELINE.apply(lambda x: 1 if x['diabmel'] == 1 and x['diabtyp'] == 2 else 0, axis=1)
    #Remove the diabtyp column
    del(df_BASELINE["diabtyp"])
    
    
    #There is one patient (101098) with missing stdyoatdt which can be replaced with his/her dtbas date
    df_BASELINE["stdyoatdt"] = df_BASELINE.apply(lambda x : x["dtbas"] if pd.isna(x["stdyoatdt"]) else x["stdyoatdt"], axis=1)


    # years_since_VTE_index: Amount of years since index VTE to the baseline visit (dtbas - vteindxdt)
    # years_since_anticoag: Amount of years since start of anticoagulant to the baseline visit (dtbas - stdyoatdt)
    df_BASELINE["years-since-VTE-index"] = df_BASELINE.apply(lambda x: round(pd.Timedelta(x["dtbas"] - x["vteindxdt"]).days/365., 2), axis=1)
    df_BASELINE["years-since-anticoag"] = df_BASELINE.apply(lambda x: round(pd.Timedelta(x["dtbas"] - x["stdyoatdt"]).days/365., 2), axis=1)
    
    #There are 13 patients where the start of anticoagulation occurred after the baseline visit, 
    #or the index VTE occurred after baseline visit which according to PWells are errors. So, we correct them.
    df_BASELINE.loc[df_BASELINE["years-since-VTE-index"]<0, "years-since-VTE-index"] = 0
    df_BASELINE.loc[df_BASELINE["years-since-anticoag"]<0, "years-since-anticoag"] = 0


    #Filling NAs for prevvtenone, prevvteprov, prevvteunprov
    #One patient could have both previous provoked and unprovoked DVT
    #If previous VTE is 0, both 'provoked' and 'unprovoked' should be 0.
    #The one person who doesn't have info for either provoked or unprovoked, its prevvtenone should be 0
    df_BASELINE.loc[(df_BASELINE["prevvtenone"]==1)&(pd.isna(df_BASELINE["prevvteprov"]))&(pd.isna(df_BASELINE["prevvteunprov"])), ["prevvtenone"]]= 0 
    #Logic: All the people with positive prevvtenone, can have prevvteprov or prevvteunprov or both 1. 
    df_BASELINE['prevvteprov'] = df_BASELINE.apply(lambda x: 1 if x['prevvtenone'] == 1 and x['prevvteprov'] == 1 else 0, axis=1)
    df_BASELINE['prevvteunprov'] = df_BASELINE.apply(lambda x: 1 if x['prevvtenone'] == 1 and x['prevvteunprov'] == 1 else 0, axis=1)


    #Any patient with missing "crbas" will be 1, otherwise, it will be 0.
    df_BASELINE["nocrbas"] = df_BASELINE.apply(lambda x: 1 if pd.isna(x["crbas"]) else 0, axis=1)

    #Any patient with missing "hemglbas" will be 1, otherwise, it will be 0.
    df_BASELINE["nohemglbbas"] = df_BASELINE.apply(lambda x: 1 if pd.isna(x["hemglbas"]) else 0, axis=1)

    #Any patients with missing "inrbas" will be 1, otherwise it will be 0.
    df_BASELINE["noinrbas"] = df_BASELINE.apply(lambda x: 1 if pd.isna(x["inrbas"]) else 0, axis=1)
    
    #Filling the 5 NAs in the vteindxdvtleg by using the info from vteindxdvtlt and vteindxdvtrt
    df_BASELINE["vteindxdvtleg"] = df_BASELINE.apply(lambda x:1 if pd.isna(x["vteindxdvtleg"]) &
                                        ((x["vteindxdvtlt"]==1)|(x["vteindxdvtrt"]==1)) else x["vteindxdvtleg"], 
                                         axis =1)
    df_BASELINE["vteindxdvtleg"] =  df_BASELINE["vteindxdvtleg"].fillna(0)
    #Remove unnecessary columns
    del(df_BASELINE["vteindxdvtlt"])
    del(df_BASELINE["vteindxdvtrt"])


    #Remove the incorrectly entered values in the Baseline dataset
    df_BASELINE = remove_incorrect_values(df_BASELINE, instructions)

    #Fill the NAs in the dataset according to the instruciton csv
    df_BASELINE = fill_na_in_df(instructions, CRF_name="BASELINE", df_data=df_BASELINE)

    #Clip values for some columns
    df_BASELINE['crbas'] = np.clip(df_BASELINE['crbas'], None, 225) #PWells says crbas >225 are correct, but I still clip them for ML models
    df_BASELINE['hemglbas'] = np.clip(df_BASELINE['hemglbas'], 75, None)
    df_BASELINE['htbascm'] = np.clip(df_BASELINE['htbascm'], 125, 225)

    #Let's add a BMI column
    df_BASELINE["bmi"] = round(df_BASELINE["wtbaskg"]/(df_BASELINE["htbascm"]/100.0)**2, 2)

    #One the encoding dic to convert integers to strings for the columns that will be one-hot-encoded
    df_BASELINE = one_hot_encode(instructions, df_BASELINE)


    #Fixing the columns with name_0 which were generated when one-hot encoded
    del(df_BASELINE["gibldbxrec_0"])
    del(df_BASELINE["cvahxrec_0"])
    del(df_BASELINE["mihxrec_0"])


    return df_BASELINE

def read_format_rawdata(data_dir, instruction_dir, discrepency_dir):
    """Reads, format, and prepare all the raw data.

    Args:
        data_dir (string): Path to the raw data excel file.
        instruction_dir (string): Path to the instructions excel file.
        discrepency_dir (string): Path to the discrepency excel file.

    Returns:
        pandas.df, pandas.df, pandas.df, pandas.df, pandas.df, pandas.df: df_FUPPREDICTOR, df_FUPOUTCOME, df_BASELINE, df_AD1, df_FUPDISCONT, df_GENOTYPE
    """
    df_FUPPREDICTOR = prepare_FUPPREDICTOR(data_dir, instruction_dir, discrepency_dir)
    df_FUPOUTCOME = prepare_FUPOUTCOME(data_dir, instruction_dir, discrepency_dir)
    df_BASELINE = prepare_BASELINE(data_dir, instruction_dir, discrepency_dir)
    df_AD1 = prepare_AD1(data_dir)
    df_FUPDISCONT = prepare_FUPDISCONT(data_dir, instruction_dir)
    df_GENOTYPE = prepare_GENOTYPE(data_dir, instruction_dir)
    
    return df_FUPPREDICTOR, df_FUPOUTCOME, df_BASELINE, df_AD1, df_FUPDISCONT, df_GENOTYPE

def prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir):
    """Create a Dataset object from all the data.

    Args:
        data_dir (string): Path to the raw data excel file.
        instruction_dir (string): Path to the instructions excel file.
        discrepency_dir (string): Path to the discrepency excel file.

    Returns:
        Dataset: The Dataset object with all the patients.
    """
    
    #Read the raw data, format and fill_NAs appropriately
    df_FUPPREDICTOR, df_FUPOUTCOME, df_BASELINE, df_AD1, df_FUPDISCONT, df_GENOTYPE = read_format_rawdata(data_dir, instruction_dir, discrepency_dir)
    
    #Create an empty Patient Dataset object
    patient_dataset = Dataset("Bleeding Patient Dataset")


    #Fill-in follow-up predictors
    for uniqid in set(df_FUPPREDICTOR["uniqid"]):
        data_FUPPREDICTOR = df_FUPPREDICTOR[df_FUPPREDICTOR["uniqid"]==uniqid].sort_values(by="fudt")
        patient_dataset.add_data_to_patient(uniqid, "FUPPREDICTOR", data_FUPPREDICTOR)
        
    #Fill-in the follow-up outcome dates
    for uniqid in set(df_FUPOUTCOME["uniqid"]):
        data_FUPOUTCOME = df_FUPOUTCOME[df_FUPOUTCOME["uniqid"]==uniqid].sort_values(by="fuodt")
        patient_dataset.add_data_to_patient(uniqid, "FUPOUTCOME", data_FUPOUTCOME)
        
    #Fill-in Baseline visit date for all_patients
    for uniqid in set(df_BASELINE["uniqid"]):
        data_BASELINE = df_BASELINE[df_BASELINE["uniqid"] == uniqid]
        patient_dataset.add_data_to_patient(uniqid, "BASELINE", data_BASELINE)
        
    #Fill-in the diagnosis date and  for each patients with follow-up
    for uniqid in set(df_AD1["uniqid"]):
        AD_1_info = df_AD1[df_AD1["uniqid"] == uniqid][["blddtadj", "majbldconf", "relbldconf", "nobld"]].to_dict(orient="records")
        patient_dataset.add_data_to_patient(uniqid, "AD1", AD_1_info)
        
    #Fill-in FUPDISCONT data
    for uniqid in set(df_FUPDISCONT["uniqid"]):
        data_FUPDISCONT = df_FUPDISCONT[df_FUPDISCONT["uniqid"] == uniqid]
        patient_dataset.add_data_to_patient(uniqid, "FUPDISCONT", data_FUPDISCONT)
        
    #Fill-in Genotype data
    for uniqid in set(df_GENOTYPE["uniqid"]):
        data_GENOTYPE = df_GENOTYPE[df_GENOTYPE["uniqid"] == uniqid]
        patient_dataset.add_data_to_patient(uniqid, "GENOTYPE", data_GENOTYPE)
        
        
    return patient_dataset

    
def get_formatted_Baseline_FUP(mode):
    """Retrieves the Baseline and FUP dataset from raw data for ML analysis.

    Args:
        mode (str): "Raw": keep all the baseline features and  slightly modify their names (for predefined Clinical models)
                    "Formatted": Remove the baseline features with more than 10% unknown, and format their names (for new ML models ).

    Returns:
        (Dataset, dict, list, pd.DataFrame, pd.Series): patient_dataset, FUPS_dict, list_FUP_cols, baseline_dataset, target_series
    """
    def format_FUP_names(list_FUP_cols):
        #Create a dictionary to convert abbreviations to long names
        FUPPREDICTOR_dic = get_abb_to_long_dic(instruction_dir, CRF_name="FUPPREDICTOR")
        FUPOUTCOME_dic = get_abb_to_long_dic(instruction_dir, CRF_name="FUPOUTCOME")

        abb_dict = dict(FUPPREDICTOR_dic)
        abb_dict.update(FUPOUTCOME_dic.items())

        #New list of column names
        new_FUP_col_list = []

        abb_dict["years-since-baseline-visit"] = "Years since baseline visit"

        #Getting rid of the Follow-up and also change the names of the columns to their long format name
        for col in list_FUP_cols:
            if "_" in col:
                if col.split("_")[1] in ["Yes", "No", "Continue", "New"]:
                    new_FUP_col_list.append(abb_dict[col.split("_")[0]].replace(" Follow-up", '')+f" ({col.split('_')[1]})")
                else:
                    new_FUP_col_list.append(abb_dict[col.split("_")[0]].replace(" Follow-up", ''))
            else:
                if col in abb_dict:
                    new_FUP_col_list.append(abb_dict[col].replace(" Follow-up", ''))
                else:
                    new_FUP_col_list.append(col)
        
        return new_FUP_col_list

    
    #Function to get the full (descriptive) name of the baseline features
    def get_long_name(abb_to_long_dictionary, col):
        def place_space_before_capitalized_str(sentence):
            new_string = ""
            for i, char in enumerate(sentence):
                if (char.isupper() & (sentence[i-1] != " ") & (~sentence[i-1].isupper()) & (~sentence[i-1].isdigit()) & (sentence[i-1] != "(") & (i!=0)):
                    new_string = new_string + " " + char
                else:
                    new_string = new_string + char
            return re.sub("\s+", " ", new_string)
        
        if "_" in col:
            if len([c for c in col if c=="_"])==1:
                return place_space_before_capitalized_str(abb_to_long_dictionary[col.split("_")[0]]+f" ({col.split('_')[1]})")
            else:
                prefix = col[0:col.rfind("_")]
                suffix = col[col.rfind("_")+1:]
                return place_space_before_capitalized_str(abb_to_long_dictionary[prefix]+f" ({suffix})")
        elif col in abb_to_long_dictionary:
            return abb_to_long_dictionary[col]
        else:
            return col
        
    #Read the patient dataset
    patient_dataset = prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir)

    #Remove patients without baseline, remove the FUPS after bleeding/termination, fill FUP data for those without FUP data
    patient_dataset.filter_patients_sequentially(mode="fill_patients_without_FUP")
    
    #Add zeroth FUP to all the patients, unless theirs were missing in the first place which were replaced with filter_patients_sequentially
    patient_dataset.add_zeroth_FUP_to_all_patients()

    #Add one feature to each patient indicating years since baseline to each FUP
    patient_dataset.add_FUP_since_baseline()
    
    print(patient_dataset.get_all_targets_counts(), "\n")

    #Get the BASELINE, and Follow-up data from patient dataset
    FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = patient_dataset.get_data_x_y(baseline_filter=["uniqid", "dtbas", "vteindxdt", "stdyoatdt"], 
                                                                                                FUP_filter=[])

    #Change the name of the ful cols to their long names
    list_FUP_cols = format_FUP_names(list_FUP_cols)
    
    #Get the genotype data
    genotype_data = patient_dataset.get_genotype_data()

    if mode == "Formatted":
        #Change the names of the columns in the baseline data
        abb_to_long_dic = get_abb_to_long_dic(instructions_dir=instruction_dir, CRF_name="BASELINE")
        abb_to_long_dic["years-since-anticoag"] = "Years since start of anticoagulation"
        abb_to_long_dic["bmi"] = "BMI"
        baseline_dataframe.columns = [get_long_name(abb_to_long_dic, col) for col in baseline_dataframe.columns]

        #Change the names of the columns in the genotype data
        abb_to_long_dic = get_abb_to_long_dic(instructions_dir=instruction_dir, CRF_name="GENOTYPE")
        genotype_data.columns = [get_long_name(abb_to_long_dic, col) for col in genotype_data.columns]
    elif mode == "Raw":
        #Change the names of the columns in the baseline data to make them compatible with the clinical score
        abb_to_long_dic = get_abb_to_long_dic(instructions_dir=instruction_dir, CRF_name="BASELINE")
        baseline_dataframe.columns = [abb_to_long_dic[col] if col in abb_to_long_dic else col for col in baseline_dataframe.columns]
    else:
        Exception(f"The mode {mode} isn't defined.")
        
    #Concat the Baseline and Genotype data
    concat_x = baseline_dataframe.join(genotype_data)
    
    ############################
    bad_features = []
    if mode == "Formatted":
        #Removing the features with more than 10% unknown
        for feature_name in concat_x.columns:
            if ("known" in feature_name):
                percent_missing = concat_x[feature_name].sum()/len(concat_x)*100
                if percent_missing > 10:
                    feature_name_prefix = feature_name.split("(")[0]
                    for feat_nam in concat_x.columns:
                        if feature_name_prefix in feat_nam:
                            bad_features.append(feat_nam)

    print("The length of features in X before removing features with many unknowns is:", len(concat_x.columns))
    
    concat_x = concat_x.drop(bad_features, axis=1)
    
    print(f" Features with many unknowns to be removed: {bad_features}")

    print("The length of features in X after removing features with many unknowns is:", len(concat_x.columns))

    return patient_dataset, FUPS_dict, list_FUP_cols, concat_x, target_series
