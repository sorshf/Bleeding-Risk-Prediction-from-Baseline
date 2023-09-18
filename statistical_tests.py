#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"This module performs all the statistics."
# =============================================================================
# Imports

from pingouin.effsize import compute_effsize
from statsmodels.stats import multitest
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel
from just_baseline_experiments import all_models
import pickle
import matplotlib.ticker as ticker



def get_grid_search_results(folder_name):
    """Parse grid search results.

    Args:
        folder_name (str): The folder in this directory containing the grid search results

    Returns:
        dic: A dictionary of where keys are models and values are the metrics measured after grid search.
    """
    
    all_model_metrics = dict()

    metrics = ["Accuracy","Precision","Sensitivity", "AUPRC", "AUROC","F1-score", "Specificity","TN","TP","FN","FP"]


    for model in all_models:
        if all_models[model] == "ML":

            model_metrics = dict()

            with open(f"./{folder_name}/{model}_nested_cv_results.pickle", 'rb') as handle:
                data = pickle.load(handle)

            for metric in metrics:
                model_metrics[metric] = list(data[f"test_{metric}"])
                
            all_model_metrics[model] = model_metrics
                
        elif all_models[model] == "Clinical":
            
            data = pd.read_csv(f"./{folder_name}/{model}_nested_cv_results.csv").drop("Unnamed: 0", axis=1).to_dict(orient='list')

            all_model_metrics[model] = data
            
    return all_model_metrics
    
def omnibus_test(all_metrics_dic, metric_name, method, alpha=0.05):
    """Performs omnibus test (Either Friedman or oneway repeated-measure ANOVA) on a dictionary of results.

    Args:
        all_metrics_dic (dic): A dictionary of where keys are models and values are the metrics measured after grid search.
        metric_name (str): The name of the metric we are testing
        method (str): Either non-parametric "Friedman", or parametric "ANOVA".
        alpha (float, optional): Alpha value for statistical tests. Defaults to 0.05.

    Returns:
        str: String description of the result of the omnibus test.
    """
    metric_values = [all_metrics_dic[model_name][metric_name] for model_name in all_metrics_dic.keys()]
    
    if method == "Friedman":
        _, p_value = friedmanchisquare(*metric_values)
    elif method == "ANOVA":
        #Get the value of the metrics for all clfs for all cv folds in order
        the_metric_values = [item for model in all_metrics_dic for item in all_metrics_dic[model][metric_name]] 

        #Number of cross validation 
        num_cv = len(list(list(all_metrics_dic.values())[0].values())[0])

        #Get the names of the clfs in order
        the_clf_names = [item for clf_name in all_metrics_dic for item in [clf_name]*num_cv]

        data_df = pd.DataFrame({
            'clf_name': the_clf_names,
            'fold_cv' : list(range(1, num_cv+1))*len(set(the_clf_names)),
            'metric_value' : the_metric_values
        })

        one_way_repeated_measure_anova = AnovaRM(data=data_df, depvar='metric_value',
                    subject='clf_name', within=['fold_cv']).fit()

        p_value = one_way_repeated_measure_anova.anova_table["Pr > F"].values[0]
    else:
        KeyError(f"The name {method} doesn't exist in fxn definition.")
    
    significance = "SIGNIFICANT" if p_value<alpha else "NOT significant"
    
    string=f"The {method} p-value is {significance} for {metric_name} across {len(metric_values)} models: {p_value:.2e}"
    
    print(string)
    
    return string

def test_omnibus_test_fxn():
    """Test the omnibus_test function.
    """
    my_dic = {"tillamook" : {"test" :[0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735,
             0.0659, 0.0923, 0.0836]},
            "newport" : {"test" :[0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835,
                    0.0725]},
            "petersburg" : {"test" :[0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]},
            "magadan" : {"test" :[0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764,
                    0.0689]},
            "tvarminne" : {"test" :[0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]}}
    
    omnibus_test(my_dic, metric_name="test", method="ANOVA", alpha=0.05)
    print("ANOVA results should be pvalue=0.0002812242314534544")
    
    print("\n")
    
    
    my_dic = {
        "before" : {"test" :[72, 96, 88, 92, 74, 76, 82]},
        "immediately_after" : {"test" :[120, 120, 132, 120, 101, 96, 112]},
        "five_min_after" : {"test" :[76, 95, 104, 96, 84, 72, 76]}
    }
    
    
    omnibus_test(my_dic, metric_name="test", method="Friedman", alpha=0.05)
    print("Friedman results should be pvalue=0.005063414171757498")
    
def calc_effect_size(all_metrics_dic, metric_name, mode):
    """Calculates the effect size (Hedges g test) of the models performance.

    Args:
        all_metrics_dic (dict): A dictionary of where keys are models and values are the metrics measured after grid search.
        metric_name (str): The metric that is being tested from the dict data.
        mode (str): "all_pairs": comapres all pairwise combinations; or "ML vs Clinical":compared ML models with Clinical

    Returns:
        pandas.df: pandas dataframe of the effect size.
    """
    #Perform pairwise effect size calculation
    if mode == "all_pairs":
        eff_size_results = pd.DataFrame(columns=all_metrics_dic.keys(), index=all_metrics_dic.keys())
    elif mode == "ML vs Clinical":
        eff_size_results = pd.DataFrame(index=[key for key in all_metrics_dic.keys() if all_models[key]=="ML"], columns=[key for key in all_metrics_dic.keys() if all_models[key]=="Clinical"])
    else:
        raise(Exception(f"{mode} isn't defined." ))   
        
    for model_1 in eff_size_results.index:
        for model_2 in eff_size_results.columns:
            if model_1 != model_2:
                ef = compute_effsize(all_metrics_dic[model_1][metric_name], 
                                            all_metrics_dic[model_2][metric_name],
                                            paired=True, eftype='hedges'
                                            )
                eff_size_results.loc[model_1, model_2] = float(ef)
            else:
                eff_size_results.loc[model_1, model_2] = 1000
                
    return eff_size_results

def calc_pairwise_p_value(all_metrics_dic, metric_name, method, mode):
    """Calculates the pairwise wilcoxon sign-rank or paired t-test on the performances of the models.

    Args:
        all_metrics_dic (dict): A dictionary of where keys are models and values are the metrics measured after grid search.
        metric_name (str): The metric that is being tested from the dict data.
        method (str): Either "Wilcoxon signed-rank test" or "Paired t-test".
        mode (str): "all_pairs": comapres all pairwise combinations; or "ML vs Clinical":compared ML models with Clinical

    Returns:
        pandas.df: pandas dataframe of the p-values.
    """
    #Perform pairwise effect size calculation
    if mode == "all_pairs":
        stat_test_results = pd.DataFrame(columns=all_metrics_dic.keys(), index=all_metrics_dic.keys())
    elif mode == "ML vs Clinical":
        stat_test_results = pd.DataFrame(index=[key for key in all_metrics_dic.keys() if all_models[key]=="ML"], columns=[key for key in all_metrics_dic.keys() if all_models[key]=="Clinical"])
       
    for model_1 in stat_test_results.index:
        for model_2 in stat_test_results.columns:
            if model_1 != model_2:
                try:
                    if method == "Wilcoxon signed-rank test":
                        statistic, p_value = wilcoxon(all_metrics_dic[model_1][metric_name], 
                                                    all_metrics_dic[model_2][metric_name],
                                                    )
                    elif method == "Paired t-test":
                        statistic, p_value = ttest_rel(all_metrics_dic[model_1][metric_name], 
                                                    all_metrics_dic[model_2][metric_name],
                                                    )
                    else:
                        KeyError(f"The method {method} does not exist.")
                                                
                except ValueError as e:
                    p_value = 1.00
                    
                stat_test_results.loc[model_1, model_2] = float(p_value)
            else:
                stat_test_results.loc[model_1, model_2] = 1000
                
    return stat_test_results

def correct_p_values(p_value_df, multitest_correction, alpha=0.05):
    """Performs p-value correction using Bonferroni method.

    Args:
        p_value_df (pandas.df): Pandas dataframe of the p-values
        multitest_correction (str): The method used to correct multiple hypothesis testing.
                                    https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
        alpha (float, optional): p-value significance level. Defaults to 0.05.

    Returns:
        corrected_stat_test_results (pd.DataFrame): Pandas dataframe of the corrected p-values.
        multitest_correction (str): The string of the type of multitest used to correct p-values.
    """
    if list(p_value_df.index) == list(p_value_df.columns):
        mode = "all_pairs"
    else:
        mode = "ML vs Clinical"
        
    
    #Create a list of ordered p-values
    ordered_p_value_lists = []
    for i in range(len(p_value_df.index)):
        for k in range(len(p_value_df.columns)):
            if mode == "all_pairs":
                if i > k:# ignore the repetitions when comparing a-->b and b-->a
                    ordered_p_value_lists.append(float(p_value_df.iloc[i,k]))
            else:
                ordered_p_value_lists.append(float(p_value_df.iloc[i,k]))

    #Correct the p-values 
    _, corrected_p_values, _, _ = multitest.multipletests(pvals=ordered_p_value_lists, alpha=alpha, 
                                                          method=multitest_correction)

    #Create a new pandas with corrected p-values
    corrected_stat_test_results = p_value_df.copy()
    index=0
    for i in range(len(corrected_stat_test_results.index)):
        for k in range(len(corrected_stat_test_results.columns)):
            if mode == "all_pairs":
                if i > k:
                    corrected_stat_test_results.iloc[i,k] = corrected_p_values[index]
                    index += 1
            else:
                corrected_stat_test_results.iloc[i,k] = corrected_p_values[index]
                index += 1
                
    return corrected_stat_test_results, multitest_correction

def plot_p_value_heatmap(p_value_df, effect_size_df, title, save_path, multitest_correction,
                         omnibus_p_value, plot_name, p_value_threshold=0.05):
    """Plot heatmap of p-values and the associated effect sizes

    Args:
        p_value_df (pandas.df): pandas dataframe of the corrected p-values.
        effect_size_df (pandas.df): pandas dataframe of the effect sizes. (Optional)
        title (str): The metric we are comparing that becomes the title of the heatmap.
        save_path (str): The path to where the figure needs to be saved.
        multitest_correction (str): The name of the multitest correction used.
        omnibus_p_value (str): The result of the ombibus test which also becomes the x-axis label.
        plot_name (str): The name of the plot to save.
        p_value_threshold (float, optional): the significance level threshold. Defaults to 0.05.
    """
    
    ignore_top_half = list(p_value_df.index) == list(p_value_df.columns)
    
        
    #Plot the hitmap for the p-values
    fig, ax = plt.subplots(figsize=(13,7))

    
    if ignore_top_half:
        #Get rid of the first row and last column, because they are all NAs
        p_value_df = p_value_df.drop(p_value_df.columns[-1], axis=1)
        p_value_df = p_value_df.drop(p_value_df.columns[0], axis=0)   
        
        if effect_size_df is not None: 
            effect_size_df = effect_size_df.drop(effect_size_df.columns[-1], axis=1)
            effect_size_df = effect_size_df.drop(effect_size_df.columns[0], axis=0)
        
        mask = np.zeros_like(p_value_df, dtype=bool)
        mask[np.triu_indices_from(mask, 1)] = True
        
    else:
        mask = np.zeros_like(p_value_df, dtype=bool)
   

    cmap = (colors.ListedColormap(['#20e84f', '#abf5bc', '#f2d666']))


    bounds = [min(p_value_df.astype('float').values.flatten())*(0.1), p_value_threshold, p_value_threshold*10, 1.01]
    
    #In case the smallest p-value was bigger than the p_value
    if min(p_value_df.astype('float').values.flatten()) > p_value_threshold:
        bounds = [p_value_threshold*(0.1), p_value_threshold, p_value_threshold*10, 1.01]
        
    #Format exponent to 10 *
    def format(value, pos=0):
        value_ = f"{value:.2e}"
        value_ =  value_.replace("e", "×10^{")
        value_ = value_ + "}"
        return f'${value_}$'
        
    
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    if effect_size_df is not None:
        labels = np.asarray([f"{value:.2e}\n({np.abs(string):.2f})" for value, string in zip(p_value_df.values.flatten(), effect_size_df.values.flatten())]).reshape(p_value_df.shape)
    else:
        labels = np.asarray([format(value) for value in p_value_df.astype(float).values.flatten()]).reshape(p_value_df.shape)
        
    
    if effect_size_df is not None:
        fontsize=6
    else:
        fontsize=8.5
        
    sns.heatmap(p_value_df.astype(float),annot=labels, cmap=cmap, norm=norm,fmt="", annot_kws={"fontsize":fontsize}, 
                square=False,linewidths=.7, ax=ax, cbar=True, cbar_kws={'label':"$\it{p}$-value", "shrink": 0.75},
                mask=mask)

    #Formnat the color bar
    colorbar = ax.collections[0].colorbar
    formatter = ticker.FuncFormatter(format)
    colorbar.ax.yaxis.set_major_formatter(formatter)
    colorbar.ax.set_position([0.67, 0.45, 0.03, 0.4])  # [left, bottom, width, height]


    ax.set_title(title, fontsize=20)
    ax.set_xlabel(f"{omnibus_p_value}\n{multitest_correction}", labelpad=10)
    
    fig.savefig(f"{save_path}{plot_name}_{multitest_correction}.pdf", bbox_inches='tight')
    
    
def main():
    metric_names = ["AUPRC", "AUROC"]
    modes = ["all_pairs","ML vs Clinical"]
    
    grid_search_results_path = "./sklearn_results/"
    stat_figure_save_path = "./sklearn_results/Figures/"

    for mode in modes:
        for metric_name in metric_names:

            all_model_metrics = get_grid_search_results(grid_search_results_path)

            omnibus_results = omnibus_test(all_model_metrics, metric_name, method="ANOVA")

            effect_df = calc_effect_size(all_model_metrics, metric_name=metric_name, mode=mode)    

            stat_df = calc_pairwise_p_value(all_model_metrics, metric_name=metric_name, method="Paired t-test", mode=mode)

            stat_df_corrected, multitest_used = correct_p_values(stat_df, multitest_correction="fdr_bh")

            plot_p_value_heatmap(stat_df_corrected, effect_size_df=effect_df, title=metric_name, 
                                 save_path = stat_figure_save_path,
                                 multitest_correction = multitest_used,
                                 plot_name=f"{metric_name}_{mode}_", 
                                 omnibus_p_value=f"{omnibus_results}", 
                                 p_value_threshold=0.05)
            
            
            
if __name__ == "__main__":
    main()