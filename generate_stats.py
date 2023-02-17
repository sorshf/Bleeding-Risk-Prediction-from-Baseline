import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import pandas as pd
from data_preparation import get_abb_to_long_dic
from constants import stats_directory
import itertools as itertools



def create_pics_for_FUPS_Cont_No_Yes(patient_dataset):
    """Generate pics for the FUPs with Continue, New, No, and Yes categories in all patients in the dataset.

    Args:
        patient_dataset (Dataset): Patient dataset.

    """
    matplotlib.use("Agg")

    SMALL_SIZE = 8

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)

    #A custom cmap to visualize zeros as black and non-zeros as green
    cmap = (colors.ListedColormap(['black', 'green'])
            .with_extremes(over='0.25', under='0.75'))

    bounds = [-1, 0.5, 1000]
    norm = colors.BoundaryNorm(bounds, cmap.N)


    #Mapping
    mapping_from_baseline_to_FUP = {
                                    "hyprthx": ['hypertfu_Continue', 'hypertfu_New', 'hypertfu_No', 'hypertfu_Yes'],
                                    "diabmel": ['diabmelfu_Continue', 'diabmelfu_New', 'diabmelfu_No', 'diabmelfu_Yes'],
                                    "atrfib": ['atrfibfu_Continue', 'atrfibfu_New','atrfibfu_No', 'atrfibfu_Yes'],
                                    "mihx": ['mifu_Continue', 'mifu_New', 'mifu_No', 'mifu_Yes'],
                                    "currpts": ['ptsfu_Continue', 'ptsfu_New','ptsfu_No', 'ptsfu_Yes'],
                                    "cvahx": ['cvafu_Continue','cvafu_No', 'cvafu_Yes'],
                                    }

    all_mapping_row = list(mapping_from_baseline_to_FUP.keys())
    all_mapping_row = ["uniqid", *all_mapping_row]

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
            
            
    for patient_set_num, patient_set in enumerate(chunks(patient_dataset.all_patients, 5)):
        fig, axs = plt.subplots(7,5, figsize=(4*5,10))
        fig_counter = 0

        for i, patient in enumerate(patient_set):
            #The plotting array is the patient's FUPOUTCOME and FUPPREDICTOR data
            plotting_array = patient.get_FUP_array().copy()
            plotting_array = pd.DataFrame(plotting_array.astype('int').to_numpy(),
                                            columns=plotting_array.columns, 
                                            index=[str(time.date()) for time in plotting_array.index])
            
            for j, feature_set in enumerate(all_mapping_row): 
                the_ax = axs[j, i]    
                
                
                #The first fig in each col is the unique id, otherwise is the figure     
                if fig_counter%7 !=0:
                    #Plot the heatmap
                    sns.heatmap(plotting_array[mapping_from_baseline_to_FUP[feature_set]].transpose(), annot=True, cmap=cmap, 
                                                norm=norm,fmt='.0f', annot_kws={"fontsize":8}, 
                                                square=True,linewidths=.1, ax=the_ax, cbar=False)
                    
                    #Currpts is a special baseline feature which has 'yes' 'no' and 'unknown' answer isntead of just 'yes' or 'no'
                    if feature_set != "currpts":
                        the_ax.set_title(f"Baseline {feature_set}: {patient.BASELINE[[feature_set]].values[0][0]}", fontsize=9)
                    else:
                        if patient.BASELINE[["currpts_No"]].values[0][0] == 1:
                            currpts_status = 0
                        elif patient.BASELINE[["currpts_Yes"]].values[0][0] == 1:
                            currpts_status = 1
                        else:
                            currpts_status = "UKN"
                            
                        the_ax.set_title(f"Baseline currpts: {currpts_status}", fontsize=9)

                        
                                        
                else:
                    #the_ax.text(0,0, s=f"{patient.uniqid}") #This is the first row of each col
                    the_ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
                    the_ax.text(s=f"{patient.uniqid}", x=0.1, y=0.25, fontdict={"size":34, "color":'red' if patient.get_target()==1 else "black"})
                    the_ax.get_xaxis().set_visible(False)
                    the_ax.get_yaxis().set_visible(False)
                    
                #The bottom plot should have the y-axis
                if j!=6:
                    the_ax.get_xaxis().set_visible(False)
                
                fig_counter += 1
                
        fig.subplots_adjust(hspace=.5)
        
        print(patient_set_num, "is done!")
        fig.savefig(f"./fups_visualized/{patient_set_num}_patient_num.pdf", bbox_inches='tight')
        
        plt.close()
        
def get_counts_all_zero_or_one_stats(patient_dataset, instruction_dir, return_mode="return"):
    """Counts the number of patients with a given feature either being all ones (positive) or all zeros (negative).

    Args:
        patient_dataset (Dataset): The patient datset object.
        instruction_dir (string): Path to the excel file instruction directory.
        return_mode (string - optional): whether to "save" or "return" the df. Default to "return".

    Raises:
        Exception: If there is patient without baseline dataset (None), an exception will be raised.
                    Also, wrong "return_mode" raises an exception.
    """

    freq_list = []

    count_dic = patient_dataset.get_all_targets_counts()

    for patient in patient_dataset:
        if patient.BASELINE is None:
            raise Exception("The dataset should be filtered so that all patients have Baselines and FUPS.")

    for feature in patient_dataset.all_patients[0].BASELINE.columns:
        freq_dic = {"CRF":"Baseline", 
                    "feature":feature,
                    "feature_name":None,
                    "total_num_patients_all_zero":0,
                    "num_bleeders_all_zero":0,
                    "num_non_bleeders_all_zero":0,
                    "total_num_patients_all_one":0,
                    "num_bleeders_all_one":0,
                    "num_non_bleeders_all_one":0}
        
        baseline_abb_long_dic = get_abb_to_long_dic(instruction_dir, "BASELINE")
        
        if feature.split("_")[0] in baseline_abb_long_dic:
            freq_dic["feature_name"] = baseline_abb_long_dic[feature.split("_")[0]]
        else:
            freq_dic["feature_name"] = feature
            
        if (pd.api.types.is_integer_dtype(patient_dataset.all_patients[0].BASELINE[feature])) & (feature!="uniqid"):
            for patient in patient_dataset.all_patients:
                if (patient.BASELINE[feature].values[0] == 0):
                    freq_dic["total_num_patients_all_zero"] += 1
                elif (patient.BASELINE[feature].values[0] == 1):
                    freq_dic["total_num_patients_all_one"] += 1
                
                if (patient.BASELINE[feature].values[0] == 0) & (patient.get_target() == 1):
                    freq_dic["num_bleeders_all_zero"] += 1
                elif (patient.BASELINE[feature].values[0] == 0) & (patient.get_target() == 0):
                    freq_dic["num_non_bleeders_all_zero"] += 1
                    
                if (patient.BASELINE[feature].values[0] == 1) & (patient.get_target() == 1):
                    freq_dic["num_bleeders_all_one"] += 1
                elif (patient.BASELINE[feature].values[0] == 1) & (patient.get_target() == 0):
                    freq_dic["num_non_bleeders_all_one"] += 1
                    
            freq_list.append(freq_dic)
            
            
    for feature in patient_dataset.all_patients[0].FUPPREDICTOR.columns:
        freq_dic = {"CRF":"FUPPREDICTOR", 
                    "feature":feature, 
                    "feature_name":None,
                    "total_num_patients_all_zero":0,
                    "num_bleeders_all_zero":0,
                    "num_non_bleeders_all_zero":0,
                    "total_num_patients_all_one":0,
                    "num_bleeders_all_one":0,
                    "num_non_bleeders_all_one":0}
        
        fuppredictor_abb_long_dic = get_abb_to_long_dic(instruction_dir, "FUPPREDICTOR")
        
        if feature.split("_")[0] in fuppredictor_abb_long_dic:
            freq_dic["feature_name"] = fuppredictor_abb_long_dic[feature.split("_")[0]]
        else:
            freq_dic["feature_name"] = feature
        
        if (pd.api.types.is_integer_dtype(patient_dataset.all_patients[0].FUPPREDICTOR[feature])) & (feature!="uniqid"):
            for patient in patient_dataset.all_patients:
                if (patient.FUPPREDICTOR[feature].sum() == 0):
                    freq_dic["total_num_patients_all_zero"] += 1
                elif (patient.FUPPREDICTOR[feature] == 1).all():
                    freq_dic["total_num_patients_all_one"] += 1

                
                if (patient.FUPPREDICTOR[feature].sum() == 0) & (patient.get_target() == 1):
                    freq_dic["num_bleeders_all_zero"] += 1
                elif (patient.FUPPREDICTOR[feature].sum() == 0) & (patient.get_target() == 0):
                    freq_dic["num_non_bleeders_all_zero"] += 1
                    
                if ((patient.FUPPREDICTOR[feature] == 1).all()) & (patient.get_target() == 1):
                    freq_dic["num_bleeders_all_one"] += 1
                elif ((patient.FUPPREDICTOR[feature] == 1).all()) & (patient.get_target() == 0):
                    freq_dic["num_non_bleeders_all_one"] += 1
                    
            freq_list.append(freq_dic)
            

    for feature in patient_dataset.all_patients[0].FUPOUTCOME.columns:
        freq_dic = {"CRF":"FUPOUTCOME", 
                    "feature":feature, 
                    "feature_name":None,
                    "total_num_patients_all_zero":0,
                    "num_bleeders_all_zero":0,
                    "num_non_bleeders_all_zero":0,
                    "total_num_patients_all_one":0,
                    "num_bleeders_all_one":0,
                    "num_non_bleeders_all_one":0}
        
        fupoutcome_abb_long_dic = get_abb_to_long_dic(instruction_dir, "FUPOUTCOME")
        
        if feature.split("_")[0] in fupoutcome_abb_long_dic:
            freq_dic["feature_name"] = fupoutcome_abb_long_dic[feature.split("_")[0]]
        else:
            freq_dic["feature_name"] = feature
        
        if (pd.api.types.is_integer_dtype(patient_dataset.all_patients[0].FUPOUTCOME[feature])) & (feature!="uniqid"):
            for patient in patient_dataset.all_patients:
                if (patient.FUPOUTCOME[feature].sum() == 0):
                    freq_dic["total_num_patients_all_zero"] += 1
                elif (patient.FUPOUTCOME[feature] == 1).all():
                    freq_dic["total_num_patients_all_one"] += 1

                
                if (patient.FUPOUTCOME[feature].sum() == 0) & (patient.get_target() == 1):
                    freq_dic["num_bleeders_all_zero"] += 1
                elif (patient.FUPOUTCOME[feature].sum() == 0) & (patient.get_target() == 0):
                    freq_dic["num_non_bleeders_all_zero"] += 1
                    
                if ((patient.FUPOUTCOME[feature] == 1).all()) & (patient.get_target() == 1):
                    freq_dic["num_bleeders_all_one"] += 1
                elif ((patient.FUPOUTCOME[feature] == 1).all()) & (patient.get_target() == 0):
                    freq_dic["num_non_bleeders_all_one"] += 1
                    
            freq_list.append(freq_dic)
            
            
    freq_df = pd.DataFrame.from_records(freq_list)

    #Calculate the percentage from count values
    freq_df["percent_patients_all_zero"] = freq_df["total_num_patients_all_zero"].apply(lambda x: round(x/count_dic["total"]*100,1))
    freq_df["percent_bleeders_all_zero"] = freq_df["num_bleeders_all_zero"].apply(lambda x: round(x/count_dic["bleeders"]*100, 1))
    freq_df["percent_non_bleeders_all_zero"] = freq_df["num_non_bleeders_all_zero"].apply(lambda x: round(x/count_dic["non-bleeders"]*100, 1))

    freq_df["percent_patients_all_one"] = freq_df["total_num_patients_all_one"].apply(lambda x: round(x/count_dic["total"]*100, 1))
    freq_df["percent_bleeders_all_one"] = freq_df["num_bleeders_all_one"].apply(lambda x: round(x/count_dic["bleeders"]*100, 1))
    freq_df["percent_non_bleeders_all_one"] = freq_df["num_non_bleeders_all_one"].apply(lambda x: round(x/count_dic["non-bleeders"]*100, 1))

    freq_df = freq_df[['CRF', 'feature', 'feature_name', 
                        'total_num_patients_all_zero','percent_patients_all_zero',
                        'num_bleeders_all_zero', 'percent_bleeders_all_zero',
                        'num_non_bleeders_all_zero','percent_non_bleeders_all_zero',
                        'total_num_patients_all_one', 'percent_patients_all_one',
                        'num_bleeders_all_one','percent_bleeders_all_one',
                        'num_non_bleeders_all_one','percent_non_bleeders_all_one'
                    ]]
    
    if return_mode == "return":
        return freq_df
    elif return_mode == "save":
        freq_df.to_excel(f"{stats_directory}Zero_One_stats.xlsx")
    else:
        raise Exception (f"Mode {return_mode} isn't defined for this function.")

def calc_freq_NO_jumps_in_FUPPREDICTOR(patient_dataset, calc_mode, return_mode):
    """Calculates the number of patients who have no changes (or jumps from "Yes", "No", "Continue", "New") in their
    "hypertfu", "diabmelfu", "atrfibfu", "mifu", "ptsfu", "cvafu". 

    Args:
        patient_dataset (Dataset): The patient's dataset.
        calc_mode (string): "all_patients", "patients_with_more_than_one_FUP"
        return_mode (string): "return", "save"

    Raises:
        Exception: If return mode isn't defined.

    Returns:
        pandas.df: Pandas dataframe of the calculated numbers.
    """
    interesting_fup_features = ["hypertfu", "diabmelfu", "atrfibfu", "mifu", "ptsfu", "cvafu"]

    fups_stat_df = pd.DataFrame()

    for prefix in interesting_fup_features:
        for suffix in ["Yes", "No", "Continue", "New"]:
            if f"{prefix}_{suffix}" in patient_dataset.all_patients[0].FUPPREDICTOR.columns: #If the column exists
                fups_stat_df.loc[f"{prefix}_{suffix}", "total"] = 0
                fups_stat_df.loc[f"{prefix}_{suffix}", "bleeders"] = 0
                fups_stat_df.loc[f"{prefix}_{suffix}", "non-bleeders"] = 0
                
                for patient in patient_dataset:
                    #Add up all the values in the columns "Yes", "No", "Continue", "New" for that feature
                    sum_of_all_vals = patient.FUPPREDICTOR[[val for val in patient.FUPPREDICTOR.columns if prefix in val]].sum().sum()
                    
                    #If the sum of all the values in the FUPPREDICTOR is different from that of the specific prefix, it is a person with all the values being that 
                    #feature
                    if (patient.FUPPREDICTOR[f"{prefix}_{suffix}"].sum() == sum_of_all_vals):
                        if calc_mode=="all_patients":

                            fups_stat_df.loc[f"{prefix}_{suffix}", "total"] += 1

                            if patient.get_target() == 1:
                                fups_stat_df.loc[f"{prefix}_{suffix}", "bleeders"] += 1
                            else:
                                fups_stat_df.loc[f"{prefix}_{suffix}", "non-bleeders"] += 1
                                
                        elif calc_mode=="patients_with_more_than_one_FUP":
                            if (len(patient.FUPPREDICTOR[f"{prefix}_{suffix}"]) > 1):
                                fups_stat_df.loc[f"{prefix}_{suffix}", "total"] += 1

                                if patient.get_target() == 1:
                                    fups_stat_df.loc[f"{prefix}_{suffix}", "bleeders"] += 1
                                else:
                                    fups_stat_df.loc[f"{prefix}_{suffix}", "non-bleeders"] += 1

    if return_mode == "return":
        return fups_stat_df
    elif return_mode == "save":
        fups_stat_df.to_excel(f"{stats_directory}FUPS_no_yes_continue_new_{calc_mode}.xlsx")
    else:
        raise Exception (f"Mode {return_mode} isn't defined for this function.")
         
def calc_freq_jumps_in_FUPPREDICTOR(patient_dataset, calc_mode, return_mode):
    """Calculates the number of jumps from "Yes", "No", "Continue", "New" in 
        "hypertfu", "diabmelfu", "atrfibfu", "mifu", "cvafu".
        
    Args:
        patient_dataset (Dataset): Patient dataset object.
        calc_mode (string): "baseline==0": only counts patients where the feature is zero in baseline.
                            "any_baseline": counts all patient regardless of their baseline value.
        return_mode (string): "return", "save".

    Raises:
        Exception: If return mode isn't defined.

    Returns:
        pandas.df: Pandas dataframe of the calculated numbers.
    """

    mapping_FUP_to_bsln = {
                            "hypertfu":"hyprthx",
                            "diabmelfu":"diabmel",
                            "atrfibfu":"atrfib",
                            "mifu":"mihx",
                            "cvafu":"cvahx"
                            }

    #Then check if first 'to' comes before the latter
    def check_from_to_fup(df, from_, to_,):
        if (df[from_].sum() > 0) and (df[to_].sum() > 0):
            return (df.loc[df[from_]==1, from_].index[0]) < (df.loc[df[to_]==1, to_].index[0])
        else:
            return False


    #Calculate the frequency of the jumps in the FUPS "Yes", "No", "Continue", "New"
    interesting_fup_features = ["hypertfu", "diabmelfu", "atrfibfu", "mifu", "cvafu"]

    suffix_lists = ["Yes", "No", "Continue", "New"]

    fups_stat_df = pd.DataFrame()

    for feature in interesting_fup_features:
        for comb in itertools.combinations(suffix_lists, 2):
            for from_, to_ in [[comb[0], comb[1]], [comb[1], comb[0]]]:
                fups_stat_df.loc[feature, f"{from_}->{to_}_total"] = 0 #Initialize the fups_stat_df
                fups_stat_df.loc[feature, f"{from_}->{to_}_bleeders"] = 0 #Initialize the fups_stat_df
                fups_stat_df.loc[feature, f"{from_}->{to_}_non-bleeders"] = 0 #Initialize the fups_stat_df
                for patient in patient_dataset:
                    if (len(patient.FUPPREDICTOR) > 1):
                        if calc_mode == "baseline==0":
                            if (patient.BASELINE[mapping_FUP_to_bsln[feature]].values[0] == 0):
                            
                                df_for_one_feature = patient.FUPPREDICTOR[[val for val in patient.FUPPREDICTOR.columns if feature in val]]
                                #print(patient.uniqid)
                                if (f"{feature}_{from_}" in df_for_one_feature.columns) and (f"{feature}_{to_}" in df_for_one_feature.columns):
                                    if check_from_to_fup(df=df_for_one_feature, from_= f"{feature}_{from_}", to_=f"{feature}_{to_}"):
                                        fups_stat_df.loc[feature, f"{from_}->{to_}_total"] += 1
                                        if patient.get_target() == 1:
                                            fups_stat_df.loc[feature, f"{from_}->{to_}_bleeders"] += 1
                                        else:
                                            fups_stat_df.loc[feature, f"{from_}->{to_}_non-bleeders"] += 1
                        elif calc_mode == "any_baseline":
                            df_for_one_feature = patient.FUPPREDICTOR[[val for val in patient.FUPPREDICTOR.columns if feature in val]]
                            #print(patient.uniqid)
                            if (f"{feature}_{from_}" in df_for_one_feature.columns) and (f"{feature}_{to_}" in df_for_one_feature.columns):
                                if check_from_to_fup(df=df_for_one_feature, from_= f"{feature}_{from_}", to_=f"{feature}_{to_}"):
                                    fups_stat_df.loc[feature, f"{from_}->{to_}_total"] += 1
                                    if patient.get_target() == 1:
                                        fups_stat_df.loc[feature, f"{from_}->{to_}_bleeders"] += 1
                                    else:
                                        fups_stat_df.loc[feature, f"{from_}->{to_}_non-bleeders"] += 1

                                    

    if return_mode == "return":
        return fups_stat_df
    elif return_mode == "save":
        fups_stat_df.to_excel(f"{stats_directory}Jumps_in_FUPS_patients_with_{calc_mode}.xlsx")
    else:
        raise Exception (f"Mode {return_mode} isn't defined for this function.")

def record_all_jumps_in_FUPPREDICTOR(patient_dataset, return_mode):
    """Record all the FUPPREDICTOR jumps that occur from "Yes", "No", "Continue", "New" in 
        "hypertfu", "diabmelfu", "atrfibfu", "mifu", "cvafu". 

    Args:
        patient_dataset (Dataset): Patient dataset object.
        return_mode (string): "save" or "return"

    Raises:
        Exception: If return mode isn't defined.

    Returns:
        pandas.df: Pandas dataframe of the calculated numbers.
    """

    mapping_FUP_to_bsln = {
                            "hypertfu":"hyprthx",
                            "diabmelfu":"diabmel",
                            "atrfibfu":"atrfib",
                            "mifu":"mihx",
                            "cvafu":"cvahx"
                            }

    #Then check if first 'to' comes before the latter
    def check_from_to_fup(df, from_, to_,):
        if (df[from_].sum() > 0) and (df[to_].sum() > 0):
            return (df.loc[df[from_]==1, from_].index[0]) < (df.loc[df[to_]==1, to_].index[0])
        else:
            return False


    #Calculate the frequency of the jumps in the FUPS "Yes", "No", "Continue", "New"
    interesting_fup_features = ["hypertfu", "diabmelfu", "atrfibfu", "mifu", "cvafu"]

    suffix_lists = ["Yes", "No", "Continue", "New"]

    fups_stat_df = []

    for feature in interesting_fup_features:
        for comb in itertools.combinations(suffix_lists, 2):
            for from_, to_ in [[comb[0], comb[1]], [comb[1], comb[0]]]:
                for patient in patient_dataset:
                    if (len(patient.FUPPREDICTOR) > 1):
                        df_for_one_feature = patient.FUPPREDICTOR[[val for val in patient.FUPPREDICTOR.columns if feature in val]]
                        if (f"{feature}_{from_}" in df_for_one_feature.columns) and (f"{feature}_{to_}" in df_for_one_feature.columns):
                            if check_from_to_fup(df=df_for_one_feature, from_= f"{feature}_{from_}", to_=f"{feature}_{to_}"):
                                
                                fups_stat_df.append([str(pd.Timestamp(patient.BASELINE["dtbas"].values[0]).date()), 
                                                                                feature, 
                                                                                patient.uniqid, 
                                                                                patient.BASELINE[mapping_FUP_to_bsln[feature]].values[0], 
                                                                                f"{from_}->{to_}"])

    
    fups_stat_df = pd.DataFrame(fups_stat_df, columns=["Baseline Data", "Feature", "uniqid", "Baseline Value", "Jump Name"])
    
    if return_mode == "return":
        return fups_stat_df
    elif return_mode == "save":
        fups_stat_df.to_excel(f"{stats_directory}fup_jumps_all_instances_for_pwells.xlsx")
    else:
        raise Exception (f"Mode {return_mode} isn't defined for this function.")                                
