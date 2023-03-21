from data_preparation import prepare_patient_dataset
from constants import data_dir, instruction_dir, discrepency_dir
from hypermodel_experiments import run_lastFUP_dense_experiment, run_baseline_dense_experiment, run_FUP_RNN_experiment, run_Baseline_FUP_multiinput_experiment, run_dummy_experiment
import sys

def main(experiment_name):
    
    #Read the patient dataset
    patient_dataset = prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir)

    #Remove patients without baseline, remove the FUPS after bleeding/termination, fill FUP data for those without FUP data
    patient_dataset.filter_patients_sequentially(mode="fill_patients_without_FUP")
    print(patient_dataset.get_all_targets_counts(), "\n")

    #Add one feature to each patient indicating year since baseline to each FUP
    patient_dataset.add_FUP_since_baseline()

    #Get the BASELINE, and Follow-up data from patient dataset
    FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = patient_dataset.get_data_x_y(baseline_filter=["uniqid", "dtbas", "vteindxdt", "stdyoatdt"], 
                                                                                               FUP_filter=[])
    print(f"Follow-up data has {len(FUPS_dict)} examples and {len(list_FUP_cols)} features.")
    print(f"Baseline data has {len(baseline_dataframe)} examples and {len(baseline_dataframe.columns)} features.")
    print(f"The target data has {len(target_series)} data.", "\n")
    
    if experiment_name == "LastFUP_Dense":
        print(f"Performing {experiment_name} experiment. \n")
    
        run_lastFUP_dense_experiment(model_name = "LastFUP_Dense", 
                                    directory_name = "keras_tuner_results", 
                                    metric_name = "prc", 
                                    metric_mode = "max", 
                                    metric_cv_calc_mode = "median", 
                                    baseline_dataframe = baseline_dataframe,
                                    FUPS_dict = FUPS_dict,
                                    target_series = target_series,
                                    list_FUP_cols = list_FUP_cols,
                                    patient_dataset = patient_dataset,
                                    overwrite=True
                                    )
        
    elif experiment_name == "Baseline_Dense":
        print(f"Performing {experiment_name} experiment. \n")

        run_baseline_dense_experiment(model_name = "Baseline_Dense", 
                                    directory_name = "keras_tuner_results", 
                                    metric_name = "prc", 
                                    metric_mode = "max", 
                                    metric_cv_calc_mode = "median", 
                                    baseline_dataframe = baseline_dataframe,
                                    FUPS_dict = FUPS_dict,
                                    target_series = target_series,
                                    list_FUP_cols = list_FUP_cols,
                                    patient_dataset = None,
                                    overwrite=True
                                    )

    elif experiment_name == "FUP_RNN":
        print(f"Performing {experiment_name} experiment. \n")

        run_FUP_RNN_experiment(model_name = "FUP_RNN", 
                                    directory_name = "keras_tuner_results", 
                                    metric_name = "prc", 
                                    metric_mode = "max", 
                                    metric_cv_calc_mode = "median", 
                                    baseline_dataframe = baseline_dataframe,
                                    FUPS_dict = FUPS_dict,
                                    target_series = target_series,
                                    list_FUP_cols = list_FUP_cols,
                                    patient_dataset = patient_dataset,
                                    overwrite=True)
        
        
    elif experiment_name == "FUP_Baseline_Multiinput":
        print(f"Performing {experiment_name} experiment. \n")
        
        run_Baseline_FUP_multiinput_experiment(model_name = "FUP_Baseline_Multiinput", 
                                    directory_name = "keras_tuner_results", 
                                    metric_name = "prc", 
                                    metric_mode = "max", 
                                    metric_cv_calc_mode = "median", 
                                    baseline_dataframe = baseline_dataframe,
                                    FUPS_dict = FUPS_dict,
                                    target_series = target_series,
                                    list_FUP_cols = list_FUP_cols,
                                    patient_dataset = patient_dataset,
                                    overwrite=True)
        
    elif experiment_name == "Dummy_classifiers":
        print(f"Performing {experiment_name} experiment. \n")
        
        run_dummy_experiment(model_name = "Dummy_classifiers", 
                        baseline_dataframe = baseline_dataframe,
                        FUPS_dict = FUPS_dict,
                        target_series = target_series)
    
    else:
        raise ValueError(f"There is no experiment name called {experiment_name}")
    

    
if __name__=="__main__":
    main(experiment_name = sys.argv[1])
    
    