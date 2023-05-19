import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from clinical_score import main as clinical_score_main
import json
import os as os
from venn import venn

from cross_validation import divide_into_stratified_fractions, get_X_y_from_indeces, normalize_training_validation
from hypermodel_experiments import create_feature_set_dicts_baseline_and_FUP
import copy

from data_preparation import prepare_patient_dataset
from constants import data_dir, instruction_dir, discrepency_dir, timeseries_padding_value
from tensorflow import keras
from data_preparation import get_abb_to_long_dic
import re

from mlxtend.evaluate import cochrans_q
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
from matplotlib import colors


model_dict = {
    "All_models": ["Baseline_Dense", "FUP_RNN", "LastFUP_Dense", "Ensemble","FUP_Baseline_Multiinput", 'CHAP','ACCP','RIETE','VTE-BLEED','HAS-BLED','OBRI'],
    "Clinical_models": ['CHAP','ACCP','RIETE','VTE-BLEED','HAS-BLED','OBRI'],
    "ML_models": ["Baseline_Dense", "LastFUP_Dense","FUP_RNN", "Ensemble","FUP_Baseline_Multiinput"]
}

model_paper_dic = {
    "Baseline_Dense":"Baseline-ANN",
    "LastFUP_Dense":"LastFUP-ANN",
    "FUP_RNN":"FUP-RNN",
    "Ensemble":"Ensemble",
    "FUP_Baseline_Multiinput":"Multimodal",
    'CHAP':"CHAP",
    'ACCP':'ACCP',
    'RIETE':'RIETE',
    'VTE-BLEED':'VTE-BLEED',
    'HAS-BLED':'HAS-BLED',
    'OBRI':'OBRI',
    "Random_clf": "Random-Classifier"
}

#create tf_get_auc function that calculates the roc auc and pr auc, test on the 
def get_auc_using_tf(tn_list, tp_list, fn_list, fp_list):
    """Uses tensorflow and keras to calculate the auc of ROC and PR curves from the tn, tp, fn, fp lists.

    Args:
        tp_list (list): List of number of true positives at 200 thresholds.
        tn_list (list): List of number of true negatives at 200 thresholds.
        fn_list (list): List of number of false negatives at 200 thresholds.
        fp_list (list): List of number of false positives at 200 thresholds.

    Returns:
        float, float: AUC_ROC, AUC_PR
    """
    
    m_ROC = tf.keras.metrics.AUC(curve="ROC")
    m_ROC.true_negatives = np.array(tn_list)
    m_ROC.true_positives = np.array(tp_list)
    m_ROC.false_negatives = np.array(fn_list)
    m_ROC.false_positives = np.array(fp_list)
    
    m_PR = tf.keras.metrics.AUC(curve="PR")
    m_PR.true_negatives = np.array(tn_list)
    m_PR.true_positives = np.array(tp_list)
    m_PR.false_negatives = np.array(fn_list)
    m_PR.false_positives = np.array(fp_list)
    
    return m_ROC.result().numpy(), m_PR.result().numpy()


def calc_PR_ROC_from_y_pred(y_pred, y_actual):
    """Calculates lists of tp, fp, fn, tn, precision, recall, and FPR at 200 thresholds from the prediction score of a binary prblem.
    Note that prediction score could be either probability (0 <= prob <= 1) or clinical score (score > 1).

    Args:
        y_pred (list): List of probability or clinical score calculated by the model.
        y_actual (list(int)): List of actual classes (0 or 1) that the instaces belong to.

    Returns:
        list, list, list, list, list, list, list, : Lists of tp, fp, fn, tn, precision, recall, and FPR.
    """
        
    epsilon = 1e-7
    thr_list = np.linspace(start=min(y_pred), stop=max(y_pred), num=198)
    thr_list = [-epsilon, *thr_list, max(y_pred)+epsilon]
    tp_list = [np.where((y_pred>=thr)&(y_actual==1) , 1, 0).sum().astype("float32") for thr in thr_list]
    fp_list = [np.where((y_pred>=thr)&(y_actual==0) , 1, 0).sum().astype("float32") for thr in thr_list]
    fn_list = [np.where((y_pred<thr)&(y_actual==1) , 1, 0).sum().astype("float32") for thr in thr_list]
    tn_list = [np.where((y_pred<thr)&(y_actual==0) , 1, 0).sum().astype("float32") for thr in thr_list]
    
    recall_curve = tf.math.divide_no_nan(
            tp_list,
            tf.math.add(tp_list, fn_list))

    FPR = tf.math.divide_no_nan(
            fp_list,
            tf.math.add(fp_list, tn_list))

    precision_curve = tf.math.divide_no_nan(
            tp_list,
            tf.math.add(tp_list, fp_list))
    
    return tp_list, fp_list, fn_list, tn_list, precision_curve, recall_curve, FPR



def plot_iterated_k_fold_scores(metric_name = "val_prc"):
    """Plots the 3 iterated k-fold scores data for the ML models with the best median metric_name.
    
    Args:
        metric_name (string): Which metric to be used to determine the best classifiers, and to be plotted.

    """
    all_plotting_data = []


    for name in model_dict["ML_models"]:
        if name != "Ensemble":

            cv_dic = pd.read_csv(f"./keras_tuner_results/{name}/{name}_grid_search_results.csv", index_col="trial_number")
            cv_dic["trial"] = [trial_num.split("_")[1] for trial_num in cv_dic.index]

            trial_nums = list(set([int(entry.split("_")[1]) for entry in cv_dic.index]))

            median_dict = dict()

            #For each trial, calculate themedian of the metric of interest across all the folds.
            for i in trial_nums:
                median_ = cv_dic[cv_dic.index.str.startswith(f"trial_{i}_")][metric_name].median()
                median_dict[i] = median_
                
            median_dict = {k: v for k, v in sorted(median_dict.items(), key=lambda item: item[1], reverse=True)}
            three_best_trial_nums = [str(val) for val in list(median_dict.keys())[0:3]]

            plotting_data = cv_dic[cv_dic["trial"].isin(three_best_trial_nums)][[f"{metric_name}", "trial"]]
            plotting_data["classifier"] = name
            
            #Order the trials
            plotting_data["trial"] = plotting_data["trial"].replace({three_best_trial_nums[0]:"Best architecture", 
                                                                    three_best_trial_nums[1]:"Second best architecture",
                                                                    three_best_trial_nums[2]:"Third best architectur"})
            
            plotting_data = plotting_data.sort_values(by="trial", ascending=True)
            
            all_plotting_data.append(plotting_data)

    all_plotting_data = pd.concat(all_plotting_data)
    
    fig, ax = plt.subplots(figsize=(12,6))
    g = sns.boxplot(x="classifier", y="val_prc", hue="trial",  data=all_plotting_data, palette="colorblind", ax=ax)
    sns.swarmplot(x="classifier", y="val_prc", hue="trial", dodge=True,color="black", data=all_plotting_data, alpha=0.5, ax=ax)

    g.legend_.set_title(None)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:3],labels[0:3])

    ax.set_ylabel("Area under precision-recall curve \n (Iterated 2-fold cv on validation set)")
    ax.set_xlabel("Classifiers")
    
    fig.savefig("./results_pics/iterated_k_fold_results_PR.png")
 
 
    
def plot_validations_train_test():
    """Plot and record csv the ROC and PR curve for the ML_models on the training-val data and test data.
    """
    
    #AUROC and PRAUC for each model is saved in ROC_PR_dic for csv
    ROC_PR_dic = dict()
    
    fig, ax = plt.subplots()
    
    model_names = model_dict["ML_models"]
    
    for name, color in zip(model_names, list(sns.color_palette("colorblind", len(model_names)))):
        roc_pcr_data = pd.read_pickle(f"./keras_tuner_results/{name}/{name}_train_test_results.pkl")

        for dataset in ["training","testing"]:
            fp_rate = list(roc_pcr_data.loc[dataset, "fp_rate"])
            recall_curve = list(roc_pcr_data.loc[dataset, "recall_curve"])
            auc = roc_pcr_data.loc[dataset, "auc"]
            if dataset == "testing":
                marker = "-."
                alpha = 1
            else:
                marker = "-"
                alpha = 0.5
            ROC_PR_dic[f"{model_paper_dic[name]}_{dataset}_AUROC"] = f"{auc:.3}"
            plt.plot(fp_rate, recall_curve, marker, label=f"{model_paper_dic[name]}_{dataset} {auc:.3}", color=color, alpha=alpha)
            
    plt.legend()
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Recall")
    
    fig.savefig(f"./results_pics/roc_ML_models_training_testing.pdf")
    
    fig, ax = plt.subplots()
    
    for name, color in zip(model_names, list(sns.color_palette("colorblind", len(model_names)))):
        roc_pcr_data = pd.read_pickle(f"./keras_tuner_results/{name}/{name}_train_test_results.pkl")

        for dataset in ["training","testing"]:
            precision_curve = list(roc_pcr_data.loc[dataset, "precision_curve"])
            recall_curve = list(roc_pcr_data.loc[dataset, "recall_curve"])
            prc = roc_pcr_data.loc[dataset, "prc"]
            if dataset == "testing":
                marker = "-."
                alpha = 1
            else:
                marker = "-"
                alpha = 0.5
            ROC_PR_dic[f"{model_paper_dic[name]}_{dataset}_AUPRC"] = f"{prc:.5}"
            plt.plot(recall_curve, precision_curve, marker, label=f"{model_paper_dic[name]}_{dataset} {prc:.5}", color=color, alpha=alpha)
        
    plt.legend()
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    
    plt.savefig(f"./results_pics/pr_ML_models_training_testing.pdf")
    
    
    data  = pd.DataFrame.from_dict(ROC_PR_dic, orient="index", columns=["value"])

    data["dataset"] = data.apply(lambda x: x.name.split("_")[1], axis=1)
    data["metric"] = data.apply(lambda x: x.name.split("_")[2], axis=1)
    data["model"] = data.apply(lambda x: x.name.split("_")[0], axis=1)

    data = data.pivot(index="model", columns=["metric", "dataset"], values="value")
    
    data.to_csv("./results_pics/detailed_training_testing_AUROC_AUPRC.csv")

  

def save_deatiled_metrics_test():
    
    all_data = []
      
    for model_name in model_dict["All_models"]:
        detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
            
        y_pred_classes = np.array(detailed_test_res["y_pred_classes"])
        y_pred = np.array(detailed_test_res["y_pred"])
        y_actual = np.array(detailed_test_res["y_actual"])
        tp_list, fp_list, fn_list, tn_list, precision_curve, recall_curve, FPR = calc_PR_ROC_from_y_pred(y_pred, y_actual)
        
        ROC_AUC, PR_AUC = get_auc_using_tf(tn_list, tp_list, fn_list, fp_list)
        
        tp = np.where((y_pred_classes==1)&(y_actual==1) , 1, 0).sum().astype("float32")
        fp = np.where((y_pred_classes==1)&(y_actual==0) , 1, 0).sum().astype("float32")
        fn = np.where((y_pred_classes==0)&(y_actual==1) , 1, 0).sum().astype("float32")
        tn = np.where((y_pred_classes==0)&(y_actual==0) , 1, 0).sum().astype("float32")
        
        recall = tf.math.divide_no_nan(
                tp,
                tf.math.add(tp, fn)).numpy()

        FPR = tf.math.divide_no_nan(
                fp,
                tf.math.add(fp, tn)).numpy()

        precision = tf.math.divide_no_nan(
                tp,
                tf.math.add(tp, fp)).numpy()
        
        all_data.append({
            "Name": model_paper_dic[model_name],
            "PR_AUC": round(PR_AUC, 3),
            "ROC_AUC": round(ROC_AUC, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "Specificity": round(1 - FPR, 3),
            "FPR": round(FPR, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
             
        })
        
    pd.DataFrame.from_records(all_data).to_csv("./results_pics/detailed_metrics_table_all_models.csv")
        
    
def plot_ROC_PR():
    """Plot the ROC and PR curve for all of the ML models and the clinical scores.
    """
    
    for model_set in model_dict:
        model_names = model_dict[model_set]
        
        fig, ax = plt.subplots(figsize=(6,5))


        for model_name, color in zip(model_names, list(sns.color_palette("colorblind", len(model_names)))):
            detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
            
            y_pred = np.array(detailed_test_res["y_pred"])
            y_actual = np.array(detailed_test_res["y_actual"])
            tp_list, fp_list, fn_list, tn_list, precision_curve, recall_curve, FPR = calc_PR_ROC_from_y_pred(y_pred, y_actual)
            
            ROC_AUC, _ = get_auc_using_tf(tn_list, tp_list, fn_list, fp_list)
            
            if model_name in model_dict["Clinical_models"]:
                marker = "--"
            else:
                marker = "*-"
            

            plt.plot(FPR, recall_curve, marker, label=f"{model_name} {ROC_AUC:.3}",linewidth=1.2,markersize=4, color=color)
            
        #The random classifier
        plt.plot([0,1], [0, 1], ":", color="black", label="Random_clf 0.50")


        handles, labels = plt.gca().get_legend_handles_labels()
        handle_label_obj = [(h,model_paper_dic[l.split(" ")[0]], float(l.split(" ")[1])) for h, l in zip(handles, labels)]

        handle_label_obj = sorted(handle_label_obj, key=lambda hl:hl[2], reverse=True)


        plt.legend([h[0] for h in handle_label_obj],[str(h[1])+f" ({h[2]})" for h in handle_label_obj], loc='best', title="Model (AUROC)", fancybox=True, fontsize='small')

                
        ax.set_xlabel("False Positive Rate (1-Specificity)", fontdict={"fontsize":12})
        ax.set_ylabel("Recall", fontdict={"fontsize":12})
        
        fig.savefig(f"./results_pics/roc_curve_{model_set}.png", dpi=300)
    
    #############################
    
    for model_set in model_dict:
        model_names = model_dict[model_set]

        fig, ax = plt.subplots(figsize=(6,5))

        for model_name, color in zip(model_names, list(sns.color_palette("colorblind", len(model_names)))):
            detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
            
            y_pred = np.array(detailed_test_res["y_pred"])
            y_actual = np.array(detailed_test_res["y_actual"])
            tp_list, fp_list, fn_list, tn_list, precision_curve, recall_curve, FPR = calc_PR_ROC_from_y_pred(y_pred, y_actual)
            
            _, PR_AUC = get_auc_using_tf(tn_list, tp_list, fn_list, fp_list)
            
            
            if model_name in model_dict["Clinical_models"]:
                marker = "--"
            else:
                marker = "*-"
            

            plt.plot(recall_curve, precision_curve, marker, label=f"{model_name} {PR_AUC:.3}",linewidth=1.2,markersize=4, color=color)

        
        num_positive = detailed_test_res["y_actual"].sum()
        total = len(detailed_test_res)
        pr_baseline = num_positive/total
        
        #The random classifier
        plt.plot([0, 1],[pr_baseline, pr_baseline], ":", color="black", label=f"Random_clf {pr_baseline:.3}")
        
        
        handles, labels = plt.gca().get_legend_handles_labels()
        handle_label_obj = [(h,model_paper_dic[l.split(" ")[0]], float(l.split(" ")[1])) for h, l in zip(handles, labels)]

        handle_label_obj = sorted(handle_label_obj, key=lambda hl:hl[2], reverse=True)


        plt.legend([h[0] for h in handle_label_obj],[str(h[1])+f" ({h[2]})" for h in handle_label_obj], loc='best', title="Model (AUPRC)", fancybox=True, fontsize='small')
                
        ax.set_xlabel("Recall", fontdict={"fontsize":12})
        ax.set_ylabel("Precision", fontdict={"fontsize":12})
        
        fig.savefig(f"./results_pics/pr_curve_{model_set}.png", dpi=300)


def plot_confusion_matrix():
    """Plot confusion matrix for all of the ML models and Clinical models.
    """

    for name in model_dict["All_models"]:
        
        detailed_test_res = pd.read_csv(f"./keras_tuner_results/{name}/{name}_detailed_test_results.csv")

        mosaic = [["All","All","All", "1", "2", "3", "4", "5", "6"],
                ["All","All","All", "7", "8", "9", "10", "11", "12"]]

        fig, axd = plt.subplot_mosaic(mosaic, figsize=(13, 4), layout="constrained")

        for fup_num in ["All","1", "2", "3", "4", "5", "6","7", "8", "9", "10", "11", "12"]:
            

            tp, fp, tn, fn = get_conf_matrix(test_res_df=detailed_test_res, fup_number=fup_num, mode="count")
            heatmap = [[tn,fp],
                    [fn,tp]]
            
            if fup_num != "All":
                sns.heatmap(heatmap, annot=True,linewidths=0.1,cmap=sns.color_palette("viridis", as_cmap=True), square=True, 
                            annot_kws={"size": 11}, fmt="g", ax=axd[fup_num], cbar=False)
                label_size = 8
                
            else:
                sns.heatmap(heatmap, annot=True,linewidths=0.1,cmap=sns.color_palette("viridis", as_cmap=True), square=True, 
                            annot_kws={"size": 15}, fmt="g", ax=axd[fup_num], cbar=False)
                
                
                label_size = 16
            
            axd[fup_num].tick_params(labelsize=label_size)


            axd[fup_num].set_xlabel("Predicted Label", fontdict={"fontsize":label_size})
            axd[fup_num].set_ylabel("True Label", fontdict={"fontsize":label_size})
            axd[fup_num].set_title(f"{fup_num} FUPS", size=label_size*1.4)

        fig.suptitle(model_paper_dic[name], size=30)
        fig.savefig(f"./results_pics/{name}_confusion_matrix.png", dpi=300)#, bbox_inches='tight')   


def extract_the_best_hps(number_of_best_hps):
    """Extract the best hyperparameters from the trained ML models.
    
    Args:
        number_of_best_hps(int): Number of best hps to save as csv.
    """

    for model_name in model_dict["ML_models"]:
        if model_name != "Ensemble":

            all_trials = []
            for trial in os.listdir(f"./keras_tuner_results/{model_name}/{model_name}/"):
                if os.path.isdir(f"./keras_tuner_results/{model_name}/{model_name}/{trial}"):
                    with open(f"./keras_tuner_results/{model_name}/{model_name}/{trial}/trial.json", 'rb') as file:
                        data = json.load(file)
                        score = data["score"]
                        data = data["hyperparameters"]["values"]
                        data["trial"] = int(trial.split("_")[1])
                        data["score"] = score
                        all_trials.append(data)
                        
            all_trials = pd.DataFrame.from_records(all_trials)
            
            all_trials.sort_values(by="score", ascending=False)[0:number_of_best_hps].to_csv(f"./results_pics/{model_name}_top_{number_of_best_hps}_hps.csv")


def get_conf_matrix(test_res_df, fup_number, mode):
    """Use the test result table (with y_actual and y_pred_classes for each instance), calculated number of 
    tp, tn, fp, and tn.

    Args:
        test_res_df (pandas.df): Pandas dataframe with y_actual and y_pred_classes for each instance of the test set.
        fup_number (string): Either "All" which returns all of the [tp, tn, fp, tn] regardless of the number of FUP for that instance,
                            or a "int" which returns [tp, tn, fp, tn] for instances with "int" number of FUPs.
        mode (string): Either "count" which returns number of [tp, tn, fp, tn], or "values" which returns the dataframe subset of [tp, tn, fp, tn].

    Returns:
        _type_: _description_
    """
    if fup_number == "All":
        tp = test_res_df[(test_res_df["y_actual"]==1)&(test_res_df["y_pred_classes"]==1)]
        tn = test_res_df[(test_res_df["y_actual"]==0)&(test_res_df["y_pred_classes"]==0)]
        fp = test_res_df[(test_res_df["y_actual"]==0)&(test_res_df["y_pred_classes"]==1)]
        fn = test_res_df[(test_res_df["y_actual"]==1)&(test_res_df["y_pred_classes"]==0)]
    else:
        tp = test_res_df[(test_res_df["FUP_numbers"]==int(fup_number))&(test_res_df["y_actual"]==1)&(test_res_df["y_pred_classes"]==1)]
        tn = test_res_df[(test_res_df["FUP_numbers"]==int(fup_number))&(test_res_df["y_actual"]==0)&(test_res_df["y_pred_classes"]==0)]
        fp = test_res_df[(test_res_df["FUP_numbers"]==int(fup_number))&(test_res_df["y_actual"]==0)&(test_res_df["y_pred_classes"]==1)]
        fn = test_res_df[(test_res_df["FUP_numbers"]==int(fup_number))&(test_res_df["y_actual"]==1)&(test_res_df["y_pred_classes"]==0)]
    
    if mode == "count":
        return len(tp), len(fp), len(tn), len(fn)
    elif mode == "values":
        return tp, fp, tn, fn
        

def get_tn_fp_fn_tn():
    """Plot the Venn diagram and saves detailed csv between tp or fn of Baseline Dense and FUP_RNN, and 
    tn fp of the Baseline Dense and FUP_RNN.
    """
    Baseline_model_detailed_test_res = pd.read_csv(f"./keras_tuner_results/Baseline_Dense/Baseline_Dense_detailed_test_results.csv")
    FUP_RNN_model_detailed_test_res = pd.read_csv(f"./keras_tuner_results/FUP_RNN/FUP_RNN_detailed_test_results.csv")

    tp_baseline, fp_baseline, tn_baseline, fn_baseline = get_conf_matrix(Baseline_model_detailed_test_res, fup_number="All", mode="values")
    tp_FUP, fp_FUP, tn_FUP, fn_FUP = get_conf_matrix(FUP_RNN_model_detailed_test_res, fup_number="All", mode="values")
    
    
    #######
    #Draw tp fn data
    fig, ax = plt.subplots(figsize=(7,7))

    venn({
        "tp_Baseline-Dense": set(tp_baseline.uniqid),
        "tp_FUP-RNN": set(tp_FUP.uniqid),
        "fn_Baseline-Dense": set(fn_baseline.uniqid),
        "fn_FUP-RNN": set(fn_FUP.uniqid),

    }, ax=ax)

    fig.savefig("./results_pics/tp_fn_on_test_set.png")
    
    ######
    
    tp_fn_dict = {"Positive_both_got_correct": set(tp_baseline.uniqid).intersection(set(tp_FUP.uniqid)),
              "Positive_only_baseline_got_correct": set(tp_baseline.uniqid).intersection(set(fn_FUP.uniqid)),
              "Positive_only_FUP_got_correct": set(tp_FUP.uniqid).intersection(set(fn_baseline.uniqid)),
              "Positive_both_got_wrong": set(fn_FUP.uniqid).intersection(set(fn_baseline.uniqid))}
    

    df_positive = []
    for val in tp_fn_dict:
        df_positive.append(pd.DataFrame.from_dict({"uniqid":list(tp_fn_dict[val]), 
                            "condition": val}))
        
    df_positive = pd.concat(df_positive)

    df_positive["prob_Baseline_Dense"] = df_positive.apply(lambda x:Baseline_model_detailed_test_res.loc[Baseline_model_detailed_test_res["uniqid"]==x["uniqid"],
                                                                                                    "y_pred"].values[0], axis=1)

    df_positive["prob_FUP_RNN"] = df_positive.apply(lambda x:FUP_RNN_model_detailed_test_res.loc[FUP_RNN_model_detailed_test_res["uniqid"]==x["uniqid"],
                                                                                                    "y_pred"].values[0], axis=1)

    
    df_positive.to_csv("./results_pics/tp_fn_on_test_set.csv", index=False)

    ##############
    
    fig, ax = plt.subplots(figsize=(7,7))

    venn({
        "tn_Baseline-Dense": set(tn_baseline.uniqid),
        "tn_FUP-RNN": set(tn_FUP.uniqid),
        "fp_Baseline-Dense": set(fp_baseline.uniqid),
        "fp_FUP-RNN": set(fp_FUP.uniqid),

    }, ax=ax)

    fig.savefig("./results_pics/tn_fp_on_test_set.png")
    
    ################
    
    tn_fp_dict = {"Negative_both_got_correct": set(tn_baseline.uniqid).intersection(set(tn_FUP.uniqid)),
              "Negative_only_baseline_got_correct": set(tn_baseline.uniqid).intersection(set(fp_FUP.uniqid)),
              "Negative_only_FUP_got_correct": set(tn_FUP.uniqid).intersection(set(fp_baseline.uniqid)),
              "Negative_both_got_wrong": set(fp_FUP.uniqid).intersection(set(fp_baseline.uniqid))}

    df_negative = []
    for val in tn_fp_dict:
        df_negative.append(pd.DataFrame.from_dict({"uniqid":list(tn_fp_dict[val]), 
                            "condition": val}))
        
    df_negative = pd.concat(df_negative)

    df_negative["prob_Baseline_Dense"] = df_negative.apply(lambda x:Baseline_model_detailed_test_res.loc[Baseline_model_detailed_test_res["uniqid"]==x["uniqid"],
                                                                                                    "y_pred"].values[0], axis=1)

    df_negative["prob_FUP_RNN"] = df_negative.apply(lambda x:FUP_RNN_model_detailed_test_res.loc[FUP_RNN_model_detailed_test_res["uniqid"]==x["uniqid"],
                                                                                                    "y_pred"].values[0], axis=1)

    df_negative.to_csv("./results_pics/tn_fp_on_test_set.csv", index=False)
    
    
def create_feature_sets_json():
    """Generates a json with the feature sets used to train the baseline and FUP models.
    """


        
    #Read the patient dataset
    patient_dataset = prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir)

    #Remove patients without baseline, remove the FUPS after bleeding/termination, fill FUP data for those without FUP data
    patient_dataset.filter_patients_sequentially(mode="fill_patients_without_FUP")
    print(patient_dataset.get_all_targets_counts(), "\n")

    #Add one feature to each patient indicating year since baseline to each FUP
    patient_dataset.add_FUP_since_baseline()

    #Get the BASELINE, and Follow-up data from patient dataset
    FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = patient_dataset.get_data_x_y(baseline_filter=["uniqid", "dtbas", "vteindxdt", "stdyoatdt", "inrbas"], 
                                                                                                FUP_filter=[])
    print(f"Follow-up data has {len(FUPS_dict)} examples and {len(list_FUP_cols)} features.")
    print(f"Baseline data has {len(baseline_dataframe)} examples and {len(baseline_dataframe.columns)} features.")
    print(f"The target data has {len(target_series)} data.", "\n")
    

    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)


    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    patient_dataset_train_val = copy.deepcopy(patient_dataset)    #Keep only the train_val data for feature selection of the FUP data
    patient_dataset_train_val.all_patients = [patient for patient in patient_dataset_train_val.all_patients if patient.uniqid in training_val_indeces]
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset_train_val, mode="Both FUP and Baseline")

    for feature_set in feature_selection_dict["baseline_feature_sets"]:
        index = feature_selection_dict["baseline_feature_sets"][feature_set]
        
        feature_selection_dict["baseline_feature_sets"][feature_set] = [baseline_dataframe.columns[i] for i in index]
        
    for feature_set in feature_selection_dict["FUPS_feature_sets"]:
        index = feature_selection_dict["FUPS_feature_sets"][feature_set]
        
        feature_selection_dict["FUPS_feature_sets"][feature_set] = [list_FUP_cols[i] for i in index]



    with open("./keras_tuner_results/feature_sets.json", 'w') as file:
        file.write(json.dumps(feature_selection_dict))


def plot_FUP_count_density():
    """Plots the count and density diagrams for the patients' number of follow-ups.
    """
    
    #Read the patient dataset
    patient_dataset = prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir)

    #Remove patients without baseline, remove the FUPS after bleeding/termination, fill FUP data for those without FUP data
    patient_dataset.filter_patients_sequentially(mode="fill_patients_without_FUP")
    print(patient_dataset.get_all_targets_counts(), "\n")

    #Add one feature to each patient indicating year since baseline to each FUP
    patient_dataset.add_FUP_since_baseline()

    #Get the BASELINE, and Follow-up data from patient dataset
    FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = patient_dataset.get_data_x_y(baseline_filter=["uniqid", "dtbas", "vteindxdt", "stdyoatdt", "inrbas"], 
                                                                                                FUP_filter=[])

    print(f"Follow-up data has {len(FUPS_dict)} examples and {len(list_FUP_cols)} features.")
    print(f"Baseline data has {len(baseline_dataframe)} examples and {len(baseline_dataframe.columns)} features.")
    print(f"The target data has {len(target_series)} data.", "\n")
    
    
    
    bleeding_frequency = []
    non_bleeder_frequency = []

    for patient in patient_dataset:
        if patient.get_target() == 1:
            bleeding_frequency.append(len(patient.get_FUP_array()))
        else:
            non_bleeder_frequency.append(len(patient.get_FUP_array()))
            


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,3))

    color1, color2 = list(sns.color_palette("colorblind", 2))

    sns.histplot(non_bleeder_frequency, discrete=True, multiple="dodge", color=color1,
                label="Non-bleeders", common_norm=False,stat="count", ax=ax1, alpha=0.5)

    sns.histplot(bleeding_frequency, discrete=True,multiple="dodge",color=color2,
                label="Bleeders",common_norm=False, stat="count", ax=ax1,alpha=0.5)

    ax1.set_xticks(range(1,14))

    ax1.legend()

    ax1.set_xlabel("Number of Follow-ups", fontdict={"fontsize":12})
    ax1.set_ylabel("Count", fontdict={"fontsize":12})

    #########

    sns.histplot(non_bleeder_frequency, discrete=True, multiple="dodge",color=color1,
                label="Non-bleeders", common_norm=False,stat="density", ax=ax2, alpha=0.5)

    sns.histplot(bleeding_frequency, discrete=True,multiple="dodge",color=color2,
                label="Bleeders", common_norm=False, stat="density", ax=ax2, alpha=0.5)

    ax2.set_xticks(range(1,14))
    ax2.legend()

    ax2.set_xlabel("Number of Follow-ups", fontdict={"fontsize":12})
    ax2.set_ylabel("Density", fontdict={"fontsize":12})
    
    
    fig.savefig("./results_pics/number_FUP_count_density.pdf", transparent=True, bbox_inches='tight')



def plot_permutaion_feature_importance_RNN_FUP(number_of_permutations=100):
    """Plot permutation importance figures for the FUP_RNN model.
    """
    
    #Read the patient dataset
    patient_dataset = prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir)

    #Remove patients without baseline, remove the FUPS after bleeding/termination, fill FUP data for those without FUP data
    patient_dataset.filter_patients_sequentially(mode="fill_patients_without_FUP")
    print(patient_dataset.get_all_targets_counts(), "\n")

    #Add one feature to each patient indicating year since baseline to each FUP
    patient_dataset.add_FUP_since_baseline()

    #Get the BASELINE, and Follow-up data from patient dataset
    FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = patient_dataset.get_data_x_y(baseline_filter=["uniqid", "dtbas", "vteindxdt", "stdyoatdt", "inrbas"], 
                                                                                                FUP_filter=[])

    print(f"Follow-up data has {len(FUPS_dict)} examples and {len(list_FUP_cols)} features.")
    print(f"Baseline data has {len(baseline_dataframe)} examples and {len(baseline_dataframe.columns)} features.")
    print(f"The target data has {len(target_series)} data.", "\n")


    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)


    #Standardize the training data, and the test data.
    _ , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    _ , norm_test_fups_X, test_y = norm_test_data

    #Load the saved model
    model = keras.models.load_model("./keras_tuner_results/FUP_RNN/FUP_RNN.h5")


    #Calculate the prc and auc on the non-purturbed dataset
    test_res = model.evaluate(norm_test_fups_X, test_y, verbose=0)
    result_dic = {metric:value for metric, value in zip(model.metrics_names,test_res)}
    auroc = result_dic["auc"]
    auprc = result_dic["prc"]

    #Create a copy of the test data, vertically stack them along the time axis, then get rid of the padded sequences
    test_fups_2_dims = copy.deepcopy(norm_test_fups_X)
    test_fups_2_dims = test_fups_2_dims.reshape(754*13, 45)
    random_sampling_fups_dataset = test_fups_2_dims[np.sum(test_fups_2_dims, axis=1)!=-225.0]


    #A function to purtub time series dataset
    def purturb_timeseries(test_fups_X, column_index):
        
        purturbed_patients_list = []

        for patient in copy.deepcopy(test_fups_X):
            new_timesteps = []
            for time in patient:
                if np.sum(time) != -225.0:
                    time[column_index] = np.random.choice(random_sampling_fups_dataset[:,column_index])
                    new_timesteps.append(time)
                else:
                    new_timesteps.append(time)
                    
            purturbed_patients_list.append(new_timesteps)
            
        return np.array(purturbed_patients_list)


    #Create a dictionary to convert abbreviations to long names
    FUPPREDICTOR_dic = get_abb_to_long_dic(instructions_dir="./Raw Data/column_preprocessing_excel test_Mar13_2022.xlsx", 
                                    CRF_name="FUPPREDICTOR")
    FUPOUTCOME_dic = get_abb_to_long_dic(instructions_dir="./Raw Data/column_preprocessing_excel test_Mar13_2022.xlsx", 
                                    CRF_name="FUPOUTCOME")

    abb_dict = dict(FUPPREDICTOR_dic)
    abb_dict.update(FUPOUTCOME_dic.items())


    #Add space in the names of the features to make them look good on figures
    def turn_space_into_newline(a_string):
        spaces_list = [a.start() for a in re.finditer(" ", a_string)]
        s = list(a_string)
        if len(spaces_list) > 3:
            s[spaces_list[int(len(spaces_list)/2)-1]] = "\n"
        return "".join(s)


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
                
    new_FUP_col_list = [turn_space_into_newline(col) for col in new_FUP_col_list]



    # For each column in FUP test set
    ## For number of permutation
    ### Copy the intact FUP test set
    ### permute the column for the FUP test set
    ### record the prc and auc

    prc_permutation_results = {col:[] for col in new_FUP_col_list}
    roc_permutation_results = {col:[] for col in new_FUP_col_list}

    for col_name, col_indx in zip(new_FUP_col_list, range(len(new_FUP_col_list))):
        for _ in range(number_of_permutations): #Number of permutations
            fup_test_copy = copy.deepcopy(norm_test_fups_X)
            
            #perturb column
            purturbed_data = purturb_timeseries(fup_test_copy, column_index=col_indx)
            
            test_res = model.evaluate(purturbed_data, test_y, verbose=0)
            result_dic = {metric:value for metric, value in zip(model.metrics_names,test_res)}
            
            prc_permutation_results[col_name].append(auprc-result_dic["prc"])
            roc_permutation_results[col_name].append(auroc-result_dic["auc"])


    color2 = sns.color_palette("colorblind", 15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5))


    data_plot = pd.DataFrame.from_dict({k: v for k, v in sorted(roc_permutation_results.items(), key=lambda item: np.mean(item[1]), reverse=True)})
    data_plot = pd.concat([data_plot.iloc[:,0:5], data_plot.iloc[:,-5:]], join='outer', axis=1)
    #ax1.boxplot(data_plot, vert=False)
    sns.boxplot(data_plot, ax=ax1, orient='h', color=color2[9], showfliers = False)
    # ax1.set_yticklabels(data_plot.columns, fontsize=8)
    ax1.vlines(0,-1,45, linestyles="--", color="grey", alpha=0.5)
    ax1.set_xlabel("Change in AUROC", fontdict={"fontsize":15})
    sns.stripplot(data_plot, ax=ax1, orient='h', palette='dark:.15', marker=".", alpha=0.2)
    # ax1.set_yticklabels(ax1.get_yticklabels(),ha="center")
    ax1.set_ylabel("Predictor Variables", fontdict={"fontsize":15})


    data_plot = pd.DataFrame.from_dict({k: v for k, v in sorted(prc_permutation_results.items(), key=lambda item: np.mean(item[1]), reverse=True)})
    #ax2.boxplot(data_plot, vert=False)
    data_plot = pd.concat([data_plot.iloc[:,0:5], data_plot.iloc[:,-5 :]], join='outer', axis=1)

    sns.boxplot(data_plot, ax=ax2, orient='h',color=color2[9],showfliers = False)
    # ax2.set_yticklabels(data_plot.columns, fontsize=8)
    ax2.vlines(0,-1,45, linestyles="--", color="grey", alpha=0.5)
    ax2.set_xlabel("Change in AUPRC", fontdict={"fontsize":15})
    sns.stripplot(data_plot, ax=ax2, orient='h', palette='dark:.15', marker=".", alpha=0.2)
    # ax2.set_yticklabels(ax2.get_yticklabels(),ha="center")

    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    fig.savefig("./results_pics/feature_importance_RNN_FUP_10values.png", dpi=300, transparent=True)



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,12))


    data_plot = pd.DataFrame.from_dict({k: v for k, v in sorted(roc_permutation_results.items(), key=lambda item: np.mean(item[1]), reverse=True)})
    #ax1.boxplot(data_plot, vert=False)
    sns.boxplot(data_plot, ax=ax1, orient='h', color=color2[9], showfliers = False)
    # ax1.set_yticklabels(data_plot.columns, fontsize=8)
    ax1.vlines(0,-1,45, linestyles="--", color="grey", alpha=0.5)
    ax1.set_xlabel("Change in AUROC", fontdict={"fontsize":16})
    sns.stripplot(data_plot, ax=ax1, orient='h', palette='dark:.15', marker=".", alpha=0.2)
    # ax1.set_yticklabels(ax1.get_yticklabels(),ha="center")
    ax1.set_ylabel("Predictor Variables", fontdict={"fontsize":16})


    data_plot = pd.DataFrame.from_dict({k: v for k, v in sorted(prc_permutation_results.items(), key=lambda item: np.mean(item[1]), reverse=True)})
    #ax2.boxplot(data_plot, vert=False)
    sns.boxplot(data_plot, ax=ax2, orient='h',color=color2[9], showfliers = False)
    # ax2.set_yticklabels(data_plot.columns, fontsize=8)
    ax2.vlines(0,-1,45, linestyles="--", color="grey", alpha=0.5)
    ax2.set_xlabel("Change in AUPRC", fontdict={"fontsize":15})
    sns.stripplot(data_plot, ax=ax2, orient='h', palette='dark:.15', marker=".", alpha=0.2)
    # ax2.set_yticklabels(ax2.get_yticklabels(),ha="center")


    plt.tight_layout()

    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    fig.savefig("./results_pics/feature_importance_RNN_FUP.png", dpi=300, transparent=True)
 
 
 
def mcnemar_analysis():
    """Perform Cochran's Q test, followed by pairwise McNemar test. Generate a pic of the p-values as a heatmap"""
    
    #Create a dictionary with the predicted class of the data on the testing set
    y_pred_dict = dict()

    #Populate the dictionary made above
    for model_name in model_dict["All_models"]:
        detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
        
        y_pred = np.array(detailed_test_res["y_pred_classes"])
        y_actual = np.array(detailed_test_res["y_actual"])
        
        y_pred_dict[model_paper_dic[model_name]] = y_pred
        

    #Perform Cochrane's Q test
    q_cochrane, p_value_cochrane = cochrans_q(y_actual, 
                            y_pred_dict["Baseline-ANN"],
                            y_pred_dict["FUP-RNN"],
                            y_pred_dict["LastFUP-ANN"],
                            y_pred_dict["Ensemble"],
                            y_pred_dict["Multimodal"],
                            y_pred_dict["CHAP"],
                            y_pred_dict["ACCP"],
                            y_pred_dict["RIETE"],
                            y_pred_dict["VTE-BLEED"],
                            y_pred_dict["HAS-BLED"],
                            y_pred_dict["OBRI"],
                            )


    #Perform pairwise mcnemar's test
    stat_test_results = pd.DataFrame(columns=y_pred_dict.keys(), index=y_pred_dict.keys())
    for model_1 in y_pred_dict.keys():
        for model_2 in y_pred_dict.keys():
            chi2, p_value = mcnemar(mcnemar_table(y_actual, 
                            y_pred_dict[model_1],
                            y_pred_dict[model_2]),
                            corrected=True, exact=True)
            stat_test_results.loc[model_1, model_2] = "{:.2e}".format(p_value)
            

    #Plot the hitmap for the p-values
    fig, ax = plt.subplots(figsize=(9.5,6))

    cmap = (colors.ListedColormap(['#20e84f', '#abf5bc', '#f2d666'])
            .with_extremes(over='9e-3', under='9e-4'))

    bounds = [9.09e-50, 9.09e-04, 9.09e-03, 1.01]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    mask = np.zeros_like(stat_test_results, dtype=bool)
    mask[np.triu_indices_from(mask)] = True


    sns.heatmap(stat_test_results.astype(float),annot=True, cmap=cmap, norm=norm,fmt=".2e", annot_kws={"fontsize":9}, 
                square=False,linewidths=.7, ax=ax, cbar=True, cbar_kws={'format': '%.2e', 'label':"$\it{p}$-value", "shrink": 0.75},
                mask=mask)



    ax.set_title(f"Cochrane's Q test p-value is {p_value_cochrane:.3g}")

    plt.savefig("./results_pics/mcnemar.png", transparent=False, bbox_inches="tight", dpi=300) 
    
        
def main():
    
    # create_feature_sets_json()
    
    # clinical_score_main()
    
    # plot_iterated_k_fold_scores()
    
    # plot_validations_train_test()
    
    # plot_ROC_PR()
    
    # plot_confusion_matrix()
    
    # extract_the_best_hps(number_of_best_hps=200)
    
    # get_tn_fp_fn_tn()

    # save_deatiled_metrics_test()
    
    # plot_FUP_count_density()
    
    # plot_confusion_matrix()
    
    # plot_permutaion_feature_importance_RNN_FUP()
    
    mcnemar_analysis()
if __name__=="__main__":
    main()