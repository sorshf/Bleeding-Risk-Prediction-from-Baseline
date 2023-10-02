#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""The module contains the definitions of two custom classes Patient and Dataset."""
# =============================================================================
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from constants import COLOR_dic, SYMBOL_dic, SIZE_dic, ALPHA_dic
import datetime
import time
from scipy import stats
#import matplotlib #uncomment this when running plot_all_data()
#matplotlib.use("Agg") #uncomment this when running plot_all_data()


class Patient():
    def __init__(self, uniqid):
        if (not isinstance(uniqid, (int, np.integer))):
            raise Exception(f"Unique IDs must be an integer. {uniqid} is a type {type(uniqid)}")
        else:
            self.uniqid = uniqid
            
        self.BASELINE  = None
        self.AD1 = None
        self.FUPPREDICTOR = None
        self.FUPOUTCOME = None
        self.FUPDISCONT = None
        self.GENOTYPE = None
        self.missing_FUP = None #This tells if the FUPOUTCOME and FUPPREDICTOR were filled by code.
        
    def get_FUPPREDICTOR_dates(self):
        """Returns the follow-up dates (extracted from FUPPREDICTOR) for each patient

        Returns:
            list: list of date objects
        """
        if self.FUPPREDICTOR is None:
            return None
        else:
            return sorted(list(self.FUPPREDICTOR["fudt"]))
        
    def get_FUPOUTCOME_dates(self):
        """Returns the follow-up dates (extracted from FUPOUTCOME) for each patient

        Returns:
            list: list of date objects
        """
        if self.FUPOUTCOME is None:
            return None
        else:
            return sorted(list(self.FUPOUTCOME["fuodt"]))
        
    def extract_timeline_data_for_CRF(self, attrib_name, data, all_dates, all_colors, all_symbols, all_sizes, all_alphas):
        """Extract the dates on a given CRF, and returns it with appropriate colors, symbols, sizes, and alphas.

        Args:
            attrib_name (String): The name of the CRF.
            data (pd.DataFrame): The pandas dataframe with the data.
            all_dates (list): List of dates.
            all_colors (list): List of colors.
            all_symbols (list): List of symbols.
            all_sizes (list): List of sizes.
            all_alphas (list): List of alphas.

        Returns:
            list, list, list, list, list: Five lists of dates, colors, symbols, sizes, alphas.
        """

        if (attrib_name == "AD1"):
            if data is None:
                return all_dates, all_colors, all_symbols, all_sizes, all_alphas
            else:
                for item in data:
                    all_dates.append(item["blddtadj"])
                    if item["majbldconf"] == 1:
                        all_colors.append(COLOR_dic["majbldconf"])
                        all_symbols.append(SYMBOL_dic["majbldconf"])
                        all_sizes.append(SIZE_dic["majbldconf"])
                        all_alphas.append(ALPHA_dic["majbldconf"])
                    elif item["relbldconf"] == 1:
                        all_colors.append(COLOR_dic["relbldconf"])
                        all_symbols.append(SYMBOL_dic["relbldconf"])
                        all_sizes.append(SIZE_dic["relbldconf"])
                        all_alphas.append(ALPHA_dic["relbldconf"])
                    elif item["nobld"] == 1:
                        all_colors.append(COLOR_dic["nobld"])
                        all_symbols.append(SYMBOL_dic["nobld"])      
                        all_sizes.append(SIZE_dic["nobld"])
                        all_alphas.append(ALPHA_dic["nobld"])

            return all_dates, all_colors, all_symbols, all_sizes, all_alphas

        
        elif (attrib_name == "FUPPREDICTOR"):
            if data is None:
                return all_dates, all_colors, all_symbols, all_sizes, all_alphas
            else:
                all_dates = all_dates + sorted(list((data["fudt"])))
                all_colors = all_colors + [COLOR_dic["fups"]]*len(sorted(list((data["fudt"]))))
                all_symbols = all_symbols + [SYMBOL_dic["fups"]]*len(sorted(list((data["fudt"]))))
                all_sizes = all_sizes + [SIZE_dic["fups"]]*len(sorted(list((data["fudt"]))))

                all_alphas = all_alphas + [ALPHA_dic["fups"]]*len(sorted(list((data["fudt"]))))



                return all_dates, all_colors, all_symbols, all_sizes, all_alphas


        elif (attrib_name == "FUPDISCONT"):
            if data is None:
                return all_dates, all_colors, all_symbols, all_sizes, all_alphas
            else:
                #b = data.drop("uniqid", axis=1)
                b = data.select_dtypes(include=[np.datetime64]).dropna(axis=1).values
                if b.shape[1] != 0:
                    all_dates = all_dates+[np.min(b)]
                    all_colors = all_colors+[COLOR_dic["fups_disc"]]
                    all_symbols = all_symbols+[SYMBOL_dic["fups_disc"]]
                    all_sizes = all_sizes+[SIZE_dic["fups_disc"]]
                    all_alphas = all_alphas+[ALPHA_dic["fups_disc"]]



                    return all_dates, all_colors, all_symbols, all_sizes, all_alphas
                else:
                    return all_dates, all_colors, all_symbols, all_sizes, all_alphas

        elif (attrib_name == "BASELINE"):
            if data is None:
                return all_dates, all_colors, all_symbols, all_sizes, all_alphas
            else:
                all_dates = all_dates+[pd.Timestamp(data["dtbas"].values[0])]
                all_colors = all_colors+[COLOR_dic["baseline"]]
                all_symbols = all_symbols+[SYMBOL_dic["baseline"]]
                all_sizes = all_sizes+[SIZE_dic["baseline"]]
                all_alphas = all_alphas+[ALPHA_dic["baseline"]]




                return all_dates, all_colors, all_symbols, all_sizes, all_alphas
        
    def get_timeline_for_plotting(self):
        """Returns the dates, colors, symbols, sizes, and alphas for plotting the timeline for a patient.

        Returns:
            list, list, list, list, list: 5 lists of dates, colors, symbols, sizes, and alphas
        """
        all_dates = []
        all_colors = []
        all_symbols = []
        all_sizes = []
        all_alphas = []
        
        info_dic = ["BASELINE", "AD1", "FUPDISCONT", "FUPPREDICTOR"]
        
        #For each CRF data, get the approproate timeline info and add them to the lists.
        for crf in info_dic:
            all_dates, all_colors, all_symbols, all_sizes, all_alphas=self.extract_timeline_data_for_CRF(crf, getattr(self, crf),
                                                            all_dates, all_colors, all_symbols, all_sizes, all_alphas)
        
        return all_dates, all_colors, all_symbols, all_sizes, all_alphas

    def plot_timeline(self, ax, y):
        """Plot the timeline of a patient on an ax object.

        Args:
            ax (matplotlib ax): An axis to plot the timeline.
            y (int): The y (vertical value) to plot the timeline on the axis.
        """
        
        #A dict with the keys being the features of FUPDISCONT page
        #and the values being their long name
        discont_FUP_dic = {"lossfudt":"Lost to Follow-up",
                           "wthdrwdt":"Withdrew consent",
                           "deathdt":"Death",
                           "permdcoatdt":"Discontinued anticoag"}
        
        #Get the timeline for plotting
        timeline = self.get_timeline_for_plotting()

        dates = timeline[0]
        colors = timeline[1]
        symbols = timeline[2]
        sizes = timeline[3]
        alphas = timeline[4]

        #Plot the line in the timeline
        ax.plot(dates, [y]*len(dates), marker="", linestyle="-", zorder=-1, color="black")
        
        #Plot the dates scatter with appropriate symbols and colors and sizes
        for date, color, symbol, size, alpha in zip(dates, colors, symbols, sizes, alphas):
            ax.scatter(date, y = y, color= color, marker=f"${symbol}$", s=size, alpha=alpha)
            #This is indicating that the scatter point is a FUPDISCONT and so, we need to write the text
            if color == "black":
                disc_text = self.FUPDISCONT.select_dtypes(include=[np.datetime64]).dropna(axis=1).columns[0]
                disc_text = discont_FUP_dic[disc_text]
                ax.text(date, y-0.5, s=disc_text, fontsize="x-small")


        #Add the uniqid to the plot 150 days before the earliest date
        ax.text(np.min(dates)-datetime.timedelta(days=150), y, s=str(self.uniqid), fontsize="small")

    def plot_all_data(self, path, instruction_dir, patient_dataset):
        """Plot each patient's FUP and Baseline data on one page using a psudo-heatmap figure.
        Note: it is recommended to set matplotlib.use("Agg") when using this function to prevent memory leakage.

        Args:
            path (String): Path to where the pics generated should be saved.
            instruction_dir (String): Path to the instructions.csv file.
            patient_dataset (Dataset): The custom Dataset object.
        """
        
        def plot_baseline_on_two_axes(baseline_data, ax1, ax2):
            """Draw the 'heatmap' of the baseline data on two axis.

            Args:
                baseline_data (pandas.df): Ddataframe of the baseline dataset.
                ax1 (matplotlib.axes): The axis to draw the first half of the baseline data.
                ax2 (matplotlib.axes): The axis to draw the second half of the baseline data.
            """
            # make a color map of fixed colors
            cmap = (colors.ListedColormap(['black', 'green'])
                    .with_extremes(over='0.25', under='0.75'))

            bounds = [-1, 0.5, 1000]
            norm = colors.BoundaryNorm(bounds, cmap.N)

            a = baseline_data.copy()
            a = a.drop(['Unique ID', 'Date of Baseline visit', 'Date of diagnosis of most recent venous thromboembolism', 
                        'Start date of oral anticoagulant to be taken for this study'], axis=1).astype('int').transpose()

            for ax, row_set in zip([ax1, ax2], np.array_split(a.index, 2)):

                partial_df = a.loc[row_set,:]

                img = ax.imshow(partial_df, cmap=cmap, norm=norm, aspect=0.6)

                ax.set_yticks(np.arange(len(partial_df)), labels=partial_df.index)

                ax.spines[:].set_visible(False)
                ax.grid(which="minor", color="r", linestyle='--', linewidth=3)
                
                ax.get_xaxis().set_visible(False)

                for j in range(len(partial_df)):
                    ax.text(0, j, round(partial_df.iloc[j,0], 2), ha="center", va="center", color="w")

        def change_col_names_in_df(instruction_dir, df, df_name):
            """Renames the columns of a pandas dataframe according to the instructions excel file.

            Args:
                instruction_dir (string): The path to the instructions excel file.
                df (pandas.df): The dataframe which we want to change its columns.
                df_name (list): List of strings (BASELINE, FUPPREDICTOR, FUPOUTCOME, etc.)

            Returns:
                pandas.df: The dataframe with the changed names.
            """
            instructions = pd.read_excel(instruction_dir)
            instructions = instructions.loc[instructions["CRF_name"].isin(df_name), ["Abbreviation", "Long_name"]]
            abbreviation_dic = {row["Abbreviation"]:row["Long_name"] for i, row in instructions.iterrows()}
            
            replacement_dic = dict()
            
            for col in df.columns:
                if "_" in col:
                    first_part = col.split("_")[0]
                    second_part = col.split("_")[1]
                    if first_part in abbreviation_dic:
                        replacement_dic[col] = f"{abbreviation_dic[first_part]}_{second_part}"
                    else:
                        replacement_dic[col] = col
                else:
                    if col in abbreviation_dic:
                        replacement_dic[col] = f"{abbreviation_dic[col]}"
                    else:
                        replacement_dic[col] = col
            
            return df.rename(columns=replacement_dic)

        #A custom cmap to visualize zeros as black and non-zeros as green
        cmap = (colors.ListedColormap(['black', 'green'])
                .with_extremes(over='0.25', under='0.75'))

        bounds = [-1, 0.5, 1000]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        #The plotting array is the patient's FUPOUTCOME and FUPPREDICTOR data
        plotting_array = self.get_FUP_array().copy()

        #Change the column names of the FUP df
        plotting_array = change_col_names_in_df(instruction_dir=instruction_dir, df=plotting_array, df_name=["FUPPREDICTOR", "FUPOUTCOME"])

        #Change the index of the df so the dates are shown more nicely.
        plotting_array = pd.DataFrame(plotting_array.astype('int').to_numpy(),
                                    columns=plotting_array.columns, 
                                    index=[str(time.date()) for time in plotting_array.index])

        #Remove decimals from weights so it fits nicely in the heatmap.
        plotting_array["Current weight (Kg) Follow-up"] = round(plotting_array["Current weight (Kg) Follow-up"], 0)


        #Create a mosaic plot for all of the figures
        axd = plt.figure(constrained_layout=True,figsize = (29.7/1.7, 21/1.7)).subplot_mosaic(
            mosaic=[
                ["ID", "Timeline", "Timeline"],
                ["Timeline_legend", "Timeline", "Timeline"],
                ["Timeline_legend", "blank","blank"],
                ["baseline_1","baseline_2","Heatmap"],
                ["baseline_1", "baseline_2", "Heatmap"],
                ["baseline_1", "baseline_2", "Heatmap"]
            ],
            empty_sentinel="blank",
            gridspec_kw = {"width_ratios": [3, 3, 7],
                        "height_ratios": [0.5, 3, 0.5, 5, 5, 5]},
        )

        #Draw the heatmap
        sns.heatmap(plotting_array.transpose(),annot=True, cmap=cmap, norm=norm,fmt='.0f', annot_kws={"fontsize":8}, 
                    square=True,linewidths=.7, ax=axd["Heatmap"], cbar=False)
        axd["Heatmap"].xaxis.tick_top() 
        axd["Heatmap"].set_xticklabels(axd["Heatmap"].get_xticklabels(), rotation=90)
        self.plot_timeline(axd["Timeline"], 0.001)

        #Remove the uniqid that patient.plot_timeline() draws by default
        for txt in axd["Timeline"].texts:
            txt.set_visible(False)
            
        axd["Timeline"].spines[['right', 'top', 'left']].set_visible(False) #Remove spines
        axd["Timeline"].get_yaxis().set_visible(False)

        #Id of the patient added to the plot
        axd["ID"].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        axd["ID"].text(s=f"{self.uniqid}", x=0, y=-3, fontdict={"size":34, "color":'red' if self.get_target()==1 else "black"})
        axd["ID"].get_xaxis().set_visible(False)
        axd["ID"].get_yaxis().set_visible(False)

        #Add the legend to the timeline legend
        patient_dataset.add_custom_legend(axd["Timeline_legend"])
        axd["Timeline_legend"].legend(loc=2, prop={'size': 7})
        axd["Timeline_legend"].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        axd["Timeline_legend"].get_xaxis().set_visible(False)
        axd["Timeline_legend"].get_yaxis().set_visible(False)

        #Draw the baseline dataset (after changing its column names to appropriate names)
        baseline_data = change_col_names_in_df(instruction_dir=instruction_dir, df=self.BASELINE.copy(), df_name=["BASELINE"])
        plot_baseline_on_two_axes(baseline_data=baseline_data, ax1=axd["baseline_1"], ax2=axd["baseline_2"])


        plt.savefig(f"{path}{'Bleeder_' if self.get_target()==1 else 'Non_bleeder_'}{self.uniqid}_all_data.pdf", bbox_inches='tight')
        
        plt.close()

    def __repr__(self):
        """String of the patient object.

        Returns:
            String: String of the patient object.
        """
        id_string = f"Patient ID: {self.uniqid}\n"
        #####################################################################################
        baseline_date = pd.Timestamp(self.BASELINE["dtbas"].values[0]).date()

        baseline_date_string = f"Baseline Visit Date: {baseline_date}\n\n"
        #####################################################################################
        if (self.get_FUPPREDICTOR_dates() is None) | (self.get_FUPOUTCOME_dates() is None):
            comparison_string = "FUPPREDICTOR and/or FUPOUTCOME is/are None."
        else:
            fuppredictor_dates = set([str(i.date()) for i in self.get_FUPPREDICTOR_dates()])
            fupoutcome_dates = set([str(i.date()) for i in self.get_FUPOUTCOME_dates()])
            comparison_string = self.align_two_lists(fuppredictor_dates, fupoutcome_dates, "FUPPREDICTOR", "FUPOUTCOME")
        
        return f"{id_string}{baseline_date_string}{comparison_string}\n#######################"
    
    def align_two_lists(self, a, b, first_list_name, second_list_name):
        """Align two lists of dates and fill-in the similarities with XXXXXXX string. 

        Args:
            a (list): The first list.
            b (list): The second list.
            first_list_name (String): The name of the first list.
            second_list_name (String): The name of the second list.

        Returns:
            String: A string with the alignment of the two lists.
        """

        total = sorted(set(a|b))

        a_new = [ele if ele in a else "XXXXXXXXXX" for ele in total]
        b_new = [ele if ele in b else "XXXXXXXXXX" for ele in total]

        string = ""

        string += f"\t {first_list_name}\t{second_list_name}\n"

        for i_, (ele_a, ele_b) in enumerate(zip(a_new, b_new)):
            string +=f"\t{str(i_+1)}. {ele_a}\t{ele_b}\n"
        return string
    
    def get_FUP_array(self):
        """Returns a dataframe with both FUPPREDICTOR and FUPOUTCOME data concatenated together.

        Returns:
            pandas Dataframe: Pandas dataframe with both FUPPREDICTOR and FUPOUTCOME data.
        """
        FUP_Pred = self.FUPPREDICTOR.set_index("fudt").drop("uniqid", axis=1)
        FUP_Outc = self.FUPOUTCOME.set_index("fuodt").drop("uniqid", axis=1)
        
        concated_df = pd.concat([FUP_Pred, FUP_Outc], axis=1, join="outer")
        
        if concated_df.isnull().values.any():
            raise Exception(f"The FUPPREDICTOR and FUPOUTCOME for patient {self.uniqid} don't have the same length.")
        
        return concated_df        

    def remove_FUP_after_bleeding_disc(self):
        """Remove all the FUP data that occur after discontinuation or bleeding.
        """
        color_dic = COLOR_dic.copy()
        color_dic = {value:key for key, value in color_dic.items()}
        all_dates, all_colors, _, _, _ = self.get_timeline_for_plotting()

        #The list of bleeding dates
        bleeding_date = [dates for dates, color in zip(all_dates, all_colors) if color_dic[color] == 'majbldconf']
        
        #The list of follow-up discontinuation dates
        discont_date = [dates for dates, color in zip(all_dates, all_colors) if color_dic[color] == 'fups_disc']

        
        #If the bleeding date list isn't empty, 
        # we remove the FUPOUTCOME and FUPPREDICTOR that come after bleeding.
        if bleeding_date:
            new_FUPPREDICTOR = self.FUPPREDICTOR[self.FUPPREDICTOR['fudt'] < bleeding_date[0]].copy()
            new_FUPOUTCOME = self.FUPOUTCOME[self.FUPOUTCOME['fuodt'] < bleeding_date[0]].copy()
            if len(new_FUPPREDICTOR) != len(self.FUPPREDICTOR):
                self.FUPPREDICTOR = new_FUPPREDICTOR
                self.FUPOUTCOME = new_FUPOUTCOME
                
            
        #If the discontinuation date list isn't empty, 
        # we remove the FUPOUTCOME and FUPPREDICTOR that come after discontinuation.
        if discont_date:
            new_FUPPREDICTOR = self.FUPPREDICTOR[self.FUPPREDICTOR['fudt'] < discont_date[0]].copy()
            new_FUPOUTCOME = self.FUPOUTCOME[self.FUPOUTCOME['fuodt'] < discont_date[0]].copy()
            if len(new_FUPPREDICTOR) != len(self.FUPPREDICTOR):
                self.FUPPREDICTOR = new_FUPPREDICTOR
                self.FUPOUTCOME =  new_FUPOUTCOME
            
    def get_bleeding_date(self):
        color_dic = COLOR_dic.copy()
        color_dic = {value:key for key, value in color_dic.items()}
        all_dates, all_colors, _, _, _ = self.get_timeline_for_plotting()

        #The list of bleeding dates
        bleeding_date = [dates for dates, color in zip(all_dates, all_colors) if color_dic[color] == 'majbldconf']
        
        if bleeding_date:
            return bleeding_date[0]
        else:
            return None
        
    def get_target(self):
        """Get the bleeding target for the patient.

        Returns:
            int: 0 for patients with no bleeding, 1 for patients with bleeding
        """
        if self.AD1 is None:
            return 0 #There is no bleeding
        else:
            for item in self.AD1:
                if item["majbldconf"] == 1: #IF there is a majbld, the return will return 1, otherwise 0.
                   return 1
            return 0
        
    def get_zeroth_FUPOUTCOME(self):
        """Generates a zeroth FUP Outcome from the baseline information. This is to have an input for RNN for the patients w/o FUP.

        Args:
            patient (patient): The patient object.

        Returns:
            pd.dataframe: Dataframe of the fup outcome features.
        """
        new_FUPOUTCOME = dict() #Empty dic

        new_FUPOUTCOME["uniqid"] = self.BASELINE["uniqid"]
        new_FUPOUTCOME["fuodt"] = self.BASELINE["dtbas"]

        #Create the pandas df from dic
        new_FUPOUTCOME = pd.DataFrame.from_dict(new_FUPOUTCOME)
        
        #Add zeros to fill all the data
        new_FUPOUTCOME[['hospfu', 'surgfu', 'chgoatfu', 'stpoatfu','bldscreenres', 'vtescreenres']] = [0,0,0,0,0,0]
        
        return new_FUPOUTCOME
    
    def get_zeroth_FUPPREDICTOR(self):
        """Generates a zeroth FUP Predictor from the baseline information. This is to have an input for RNN for the patients w/o FUP.

        Args:
            patient (patient): The patient object.

        Returns:
            pd.dataframe: Dataframe of the fup outcome features.
        """
        new_FUPPREDICTOR = dict()

        new_FUPPREDICTOR["fudt"] = self.BASELINE["dtbas"]
        new_FUPPREDICTOR["cignumfu"] = self.BASELINE["cignum"]
        new_FUPPREDICTOR["nummedfu"] = self.BASELINE["nummedbas"]
        new_FUPPREDICTOR["ocpfu"] = self.BASELINE["ocpbas"]
        new_FUPPREDICTOR["estrfu"] = self.BASELINE["estrbas"]
        new_FUPPREDICTOR["platfu"] = self.BASELINE["platbas"]
        new_FUPPREDICTOR["nsaidfu"] = self.BASELINE["nsaidbas"]
        new_FUPPREDICTOR["amiofu"] = self.BASELINE["amiobas"]
        new_FUPPREDICTOR["ssrifu"] = self.BASELINE["Amiodarone"] #SSRI is in fact encoded as AMIODARONE in the dataset
        new_FUPPREDICTOR["statfu"] = self.BASELINE["statin"]
        new_FUPPREDICTOR["antibfu"] = 0 #Most other values are zero
        new_FUPPREDICTOR["wtfukg"] = self.BASELINE["wtbaskg"]
        new_FUPPREDICTOR["uniqid"] = self.BASELINE["uniqid"]

        #Create the pandas df
        new_FUPPREDICTOR = pd.DataFrame.from_dict(new_FUPPREDICTOR)

        new_FUPPREDICTOR[['hypertfu_Continue', 'hypertfu_New', 'hypertfu_No', 'hypertfu_Yes']] = [0,0,1,0]
        new_FUPPREDICTOR[['diabmelfu_Continue', 'diabmelfu_New', 'diabmelfu_No', 'diabmelfu_Yes']] = [0,0,1,0]
        new_FUPPREDICTOR[['mifu_Continue', 'mifu_New', 'mifu_No', 'mifu_Yes',]] = [0,0,1,0]
        new_FUPPREDICTOR[['cvafu_Continue','cvafu_No', 'cvafu_Yes',]] = [0,1,0]
        new_FUPPREDICTOR[['atrfibfu_Continue', 'atrfibfu_New','atrfibfu_No', 'atrfibfu_Yes']] = [0,0,1,0]
        new_FUPPREDICTOR[['ptsfu_Continue', 'ptsfu_New','ptsfu_No', 'ptsfu_Yes']] = [0,0,1,0]
        new_FUPPREDICTOR[['cancfu_Continue', 'cancfu_New', 'cancfu_No','cancfu_Yes']] = [0,0,1,0]
        
        return new_FUPPREDICTOR
        
        
    def fill_missing_fups(self):
        """Artificially fill the FUPPREDICTOR and FUPOUTCOME with a naive logic.
        """
        
        def fill_FUPOUTCOME(patient):
            #Set the FUPOUTCOME
            patient.FUPOUTCOME = self.get_zeroth_FUPOUTCOME()
            patient.missing_FUP = True
              
        def fill_FUPPREDICTOR(patient):
            #Set the FUPPREDICTOR
            patient.FUPPREDICTOR = self.get_zeroth_FUPPREDICTOR()
            patient.missing_FUP = True
                
        if (self.FUPOUTCOME is None):
            fill_FUPOUTCOME(self)
        elif (len(self.FUPOUTCOME)==0):
            fill_FUPOUTCOME(self)
        else:
            raise RuntimeError(f"The patient {self.uniqid} already has FUPOUTCOME data. We only fill FUP data for patients with missing FUP.")
                

        ###############
        
        if (self.FUPPREDICTOR is None):
            fill_FUPPREDICTOR(self)
        elif (len(self.FUPPREDICTOR)==0):
            fill_FUPPREDICTOR(self)
        else:
            raise RuntimeError(f"The patient {self.uniqid} already has FUPPREDICTOR data. We only fill FUP data for patients with missing FUP.")
            
        
class Dataset():
    
    def __init__(self, name):
        self.name = name
        self.all_patients = []
        self.missing_genotypes_filled = False
        
    def add_patient(self, patient):
        """Add a patient object to the dataset.

        Args:
            patient (Patient): A patient object.

        Raises:
            Exception: If the patient with the same ID exists, an exception will be raised.
        """
        if patient.uniqid in self.get_all_ids():
            raise Exception(f"Patient with id {patient.uniqid} is already in the dataset.")
        else:
            self.all_patients.append(patient)
        
    def __contains__(self, patient):
        return patient in self.all_patients
    
    def __str__(self) -> str:
        
        count_dic = self.get_all_targets_counts()
        
        return f"The dataset {self.name}, contains {count_dic['bleeders']} bleeders and {count_dic['non-bleeders']} non-bleeders (total of {count_dic['total']} patients)."
    
    def add_data_to_patient(self, uniqid, attr_name, attr_value):
        """If patient already exists, get the patient object and add the data.
           Otherwise create the object, add data, and add patient to the dataset.

        Args:
            uniqid (int): Uniqid of the patient.
            attr_name (string): The name of the attribute.
            attr_value (Any): The value of the attribute.
        """
        if uniqid in self.get_all_ids():
            patient = self[uniqid]
            setattr(patient, attr_name, attr_value)
        else:
            patient = Patient(uniqid)
            setattr(patient, attr_name, attr_value)
            self.add_patient(patient)
    
    def __iter__(self):
        return iter(self.all_patients)
    
    def __getitem__(self, uniqid):
        if not isinstance(uniqid, (int, np.integer)):
            raise Exception(f"Unique IDs must be an integer. {uniqid} is a type {type(uniqid)}")
        if uniqid in self.get_all_ids():
            for patient in self.all_patients:
                if patient.uniqid == uniqid:
                    return patient
        else:
            raise Exception(f"Patient with ID {uniqid} does not exist in the dataset.")
            
    def get_all_ids(self):
        return [patient.uniqid for patient in self.all_patients]
    
    def print_diff_OUTC_PRED(self, mode="print"):
        """Patients with different lengths of OUTCOME and PREDICTOR (i.e they have a missing value)
            will be printed.

        Args:
            mode (str, optional): Whether to print each patient or just return the counter number. Defaults to "print".

        Returns:
            int: The counter. 
        """
        counter = 0
        for patient in self.all_patients:
            if patient.get_FUPOUTCOME_dates() != patient.get_FUPPREDICTOR_dates():
                if mode == "print":
                    print(patient)
                else:
                    counter += 1
        if mode != "print":
            return counter
        
    def get_patients_without_baseline(self):
        """Returns lists of patients without baseline.

        Returns:
            list: Patients objects without baseline
        """
        patient_list = []
        for patient in self.all_patients:
            if patient.BASELINE is None:
                patient_list.append(patient)
        return patient_list
    
    def patients_with_diff_followups(self):
        """Returns list of patients with different number of follow-ups in FUPOUTCOME and FUPPREDICTOR.

        Returns:
            list: List of Patients.
        """
        patient_list = []
        for patient in self.all_patients:
            if (patient.FUPOUTCOME is not None) & (patient.FUPPREDICTOR is not None):
                if sorted(set(patient.FUPOUTCOME["fuodt"])) != sorted(set(patient.FUPPREDICTOR["fudt"])):
                    patient_list.append(patient)
                    
        return patient_list
    
    def patients_without_FUPs(self):
        """Returns patients that don't have FUPOUTCOME and FUPPREDICTOR.

        Returns:
            List: List of Patients.
        """
        patient_list = []
        for patient in self.all_patients:
            if (patient.FUPOUTCOME is None) & (patient.FUPPREDICTOR is None):
                patient_list.append(patient)
                
        return patient_list
    
    def filter_patients_sequentially(self, mode):
        """Removes the patients without baseline. Then depending on the mode, it either:
        1- removes those w/o FUP and those whose FUP are after bleeding/discontinuation.
        2- artificially fill in one FUP vector for patients w/o FUP and those whose FUP are after bleeding/discontinuation.

        Args:
            mode (String): The modes: 1- "remove_patients_W/O_FUP_or_Baseline_NO_filling"
                                      2- "fill_patients_without_FUP"

        Raises:
            Exception: If the mode isn't defined, an exception is raised.
        """
        
        #Remove tho patients without baseline
        #This occurs for any modes.
        patients_without_baseline = []
        new_patients_list = []
        for patient in self:
            if (patient.BASELINE is None):
                patients_without_baseline.append(patient)
            else:
                new_patients_list.append(patient)
                
        print(f"{len(patients_without_baseline)}/{len(self.all_patients)} of patients were removed because they don't have BASELINE data.")
        self.all_patients = new_patients_list
        
        if mode == "remove_patients_W/O_FUP_or_Baseline_NO_filling":
            #Remove those without FUP data
            patients_without_FUP = []
            new_patients_list = []
            for patient in self:
                if (patient.FUPPREDICTOR is None):
                    patients_without_FUP.append(patient)
                else:
                    new_patients_list.append(patient)
            
            print(f"{len(patients_without_FUP)}/{len(self.all_patients)} of patients were removed because they don't have FUP data.")
            self.all_patients = new_patients_list
            
            
            #Remove the events after each bleeding or discontinuation
            #Afterwards, if the patient's FUP were none, remove them from the dataset.
            patients_with_no_events_before_bleeding = []
            new_patients_list = []
            for patient in self:
                patient.remove_FUP_after_bleeding_disc()
                if len(patient.get_FUP_array()) == 0:
                    patients_with_no_events_before_bleeding.append(patient)
                else:
                    new_patients_list.append(patient)
            
            print(f"{len(patients_with_no_events_before_bleeding)}/{len(self.all_patients)} of patients were removed because they don't have any event before bleeding/discontinuation.")
            self.all_patients = new_patients_list
            
            
        elif mode == "fill_patients_without_FUP":
            #Add FUP to the patients without FUP data
            patients_without_FUP = 0
            for patient in self:
                if (patient.FUPPREDICTOR is None):
                    patient.fill_missing_fups()
                    patients_without_FUP +=1 

            print(f"{patients_without_FUP}/{len(self.all_patients)} of patients don't have FUP data. They were added with code.")
            
                    
            #Remove the events after each bleeding or discontinuation
            patients_with_no_events_before_bleeding = 0
            for patient in self:
                patient.remove_FUP_after_bleeding_disc()
                if len(patient.get_FUP_array()) == 0:
                    patient.fill_missing_fups() #There are 11 patients whose discontinuation occur at baseline, so that's why we need to add the artificial FUP vector again.
                    patients_with_no_events_before_bleeding += 1
            
            print(f"Added one FUP vector with code to {patients_with_no_events_before_bleeding}/{len(self.all_patients)} of patients who don't have any event before bleeding/discontinuation.")
        else:
            raise Exception(f"The mode {mode} isn't defined in filter_patients_sequentially() function.")
    
    def add_zeroth_FUP_to_all_patients(self):
        counter = 0
        for patient in self:
            if not patient.missing_FUP:
                patient.FUPOUTCOME = pd.concat([patient.get_zeroth_FUPOUTCOME(), patient.FUPOUTCOME])
                patient.FUPPREDICTOR = pd.concat([patient.get_zeroth_FUPPREDICTOR(), patient.FUPPREDICTOR])
                counter += 1
        
        num_patients_with_missing_FUPS = len([patient for patient in self if patient.missing_FUP])
        
        print(f"The zeroth FUP was added for {counter} patients. Number of patients with no actual FUPs is {num_patients_with_missing_FUPS}.")
                                 
    def sort_patients(self):
        """Sort patients in the dataset such that the patients with major bleeding are at the begining of the list.
        """
        bleeders = []
        non_bleeders = []
        for patient in self.all_patients:
            _, colors, _, _, _ = patient.extract_timeline_data_for_CRF("AD1", patient.AD1, [], [], [], [], [])
            #timeline = patient.make_timeline_rows("AD1", patient.AD1)
            #Having red in the colors means that there was a bleeding.
            if len(colors) == 0:
                non_bleeders.append(patient)
            else:
                if COLOR_dic["majbldconf"] in colors:
                    bleeders.append(patient)
                else:
                    non_bleeders.append(patient)
        self.all_patients = [*bleeders, *non_bleeders]
                
    def add_custom_legend(self, ax):
        """Add a custom legend (for the bleeding timeline) to a matplotlib axis.

        Args:
            ax (matplotlib ax): matplotlib axis object
        """
        color_dic = COLOR_dic.copy()
        symbol_dic = SYMBOL_dic.copy()
        patches = []

        label_name_dics = {'majbldconf': 'Major bleeding confirmed',
                                'relbldconf': 'Clinically relevant non-major bleeding',
                                'nobld': 'No bleeding',
                                'baseline': 'Baseline visit',
                                'fups': 'Follow-ups',
                                'fups_disc': 'Follow-up discontinuation'
                                }

        for name in label_name_dics:
            a = ax.scatter([],[], marker=f"${symbol_dic[name]}$", s=100, color=color_dic[name], label=label_name_dics[name])
            patches.append(a)

        ax.legend(handles=patches, #bbox_to_anchor=(0, 1), 
                    loc='best', ncol=1, fontsize=14,markerscale=2 ) 
                           
    def plot_timeline_multiple_pages(self, path, num_pages=13):
        """Plot the timelines for all the patients across multiple PDF pages
        
        Args:
            path (string): The path to where the pics should be saved.
            num_pages (int): The number of pdf pages that should be produced.
        """
        
        def save_a_fig(patients_set, set_number):
            fig, ax = plt.subplots(figsize=(15,35))
            for patient_number, patient in enumerate(list(patients_set)):
                patient.plot_timeline(ax, patient_number) #Plot the patient timeline
            
            ax.set_xlim(pd.Timestamp("2007-06-01"), pd.Timestamp("2018-06-01"))
            ax.xaxis.tick_top()
            for x in ax.get_xticks():
                ax.vlines(x, 0, patient_number+2, zorder=-5, color="black", linestyle="--", alpha=0.2)
            ax.set_ylim(bottom=-2, top=patient_number+2)
            ax.set_yticklabels([])

            self.add_custom_legend(ax)


            fig.tight_layout()
            ax.invert_yaxis()
            fig.savefig(f"{path}Bleeding_risk_timeline_Final_{time.time()}_{set_number}_New.pdf") 
            plt.close()
            print(f"Page {set_number+1} was saved.")
        
        
        plt.rcParams['font.family'] = "Arial"

        #Sort patients so those with bleeding are at the top of the list.
        self.sort_patients()
        
        for set_number, patients_set in enumerate(np.array_split(list(reversed(self.all_patients)), num_pages)):
            save_a_fig(patients_set, set_number)
                         
    def create_stats_for_baseline_nominal_features(self, path):
        """Create a csv file in path showing the number of zeros and ones for each column of the baseline
        dataset.

        Args:
            path (string): String to the path where the file is being saved.
        """
        all_baselines = []

        for patient in self.all_patients:
            if patient.BASELINE is not None:
                baseline_data = patient.BASELINE.copy()
                baseline_data["target"] = patient.get_target()
                all_baselines.append(baseline_data)
                
        all_baselines = pd.concat(all_baselines)

        stats_dic = dict()

        for col in all_baselines.columns:
            if (set(all_baselines[col].unique()) == set([0,1])) and (col!="target"):
                bleeders_zero = all_baselines[(all_baselines["target"]==1) & (all_baselines[col]==0)].shape[0]
                bleeders_one = all_baselines[(all_baselines["target"]==1) & (all_baselines[col]==1)].shape[0]
                non_bleeders_zero = all_baselines[(all_baselines["target"]==0) & (all_baselines[col]==0)].shape[0]
                non_bleeders_one = all_baselines[(all_baselines["target"]==0) & (all_baselines[col]==1)].shape[0]
                
                data = pd.Series([bleeders_zero,bleeders_one,non_bleeders_zero,non_bleeders_one], index=["bleeders_zero","bleeders_one","non_bleeders_zero","non_bleeders_one"])
                
                
                stats_dic[col] = data

        pd.DataFrame.from_dict(stats_dic, orient="index").to_csv(f"{path}/baseline_nominal_summary.csv")
        
    def add_FUP_since_baseline(self):
        for patient in self.all_patients:
            if patient.FUPPREDICTOR is not None:
                baseline_date = patient.BASELINE["dtbas"].values[0]
                patient.FUPPREDICTOR["years-since-baseline-visit"] = patient.FUPPREDICTOR.apply(lambda x: round(pd.Timedelta(x["fudt"] - baseline_date).days/365., 2), axis=1)
    
    def get_all_targets_counts(self):
        bleeders_count = 0
        non_bleeders_count = 0
        
        for patient in self:
            if patient.get_target() == 0:
                non_bleeders_count += 1
            else:
                bleeders_count += 1
        
        return {"bleeders":bleeders_count, "non-bleeders": non_bleeders_count, "total": len(self.all_patients)}
    
    def get_data_x_y(self, baseline_filter, FUP_filter):
        """Returns the baseline, FUP data, and target of a Dataset object.
        
        Args:
            baseline_filter (list[str]): List of feature names that shouldn't be included in the final baseline x.
            FUP_filter (list[str]): List of features names that shouldn't be included in the final FUP dataset.

        Returns:
            dict, list, pandas.df, pandas.series : FUPS_dict, list_FUP_cols, baseline_dataframe, target_series
        """
        
        FUPS_dict = dict() #A dictionary with keys:values corresponding to uniqid:FUP_numpy_arrays
        baseline_list = [] #List of all of the baseline values as numpy array
        target_list = [] #List of baseline datasets


        for patient in self.all_patients:
            baseline = patient.BASELINE.copy().to_numpy(dtype="float32").flatten() #Baseline data
            fups = patient.get_FUP_array().copy().drop(columns=FUP_filter, axis=1).to_numpy(dtype="float32") #FUP data with specific columns dropped.
            target = patient.get_target() #Target data
            
            #Append the baseline, FUP, and target to the appropriate variable for each patient.
            FUPS_dict[patient.uniqid]=fups
            baseline_list.append(baseline)
            target_list.append(target)
        
        #Using the BASELINE data above, create a pandas df and remove the unwanted features
        #Using the target list above, create a pandas series
        #Both the BASELINE df and target series are indexed using the uniqids
        baseline_dataframe = pd.DataFrame(baseline_list, columns=patient.BASELINE.columns)
        baseline_dataframe.index = baseline_dataframe["uniqid"].astype(int)
        baseline_dataframe = baseline_dataframe.drop(columns=baseline_filter, axis=1)
        target_series = pd.Series(target_list, index=baseline_dataframe.index.astype(int))    
        
        list_of_FUP_cols = list(patient.get_FUP_array().drop(columns=FUP_filter, axis=1).columns)
        
        return FUPS_dict, list_of_FUP_cols, baseline_dataframe, target_series
    
    def correct_FUPPREDICTION_with_new_columns(self):
        """For each patient, create 3 new columns in FUPS (diabmelfu, hypertfu, and atrfibfu) as PWELLS explained. 
        """
        def correct_FUPPREDICTION_New_Columns(patient_object, FUP_feature_name, BASELINE_feature_name):
            #If patients had Diabetes at baseline they always have it so the follow up stuff doesn’t matter. 
            # For the patients with No DM at baseline they can be considered to have developed diabetes when the “new” is a one
            patient_object.FUPPREDICTOR[FUP_feature_name] = 0

            if patient_object.BASELINE[BASELINE_feature_name].values[0] == 0:
                patient_object.FUPPREDICTOR[FUP_feature_name] = 0
                feature_fu = patient_object.FUPPREDICTOR[f'{FUP_feature_name.split("_")[0]}_New']
                if 1 in list(feature_fu.values):
                    index_of_diabmel = feature_fu[feature_fu==1].index[0]
                    patient_object.FUPPREDICTOR.loc[index_of_diabmel,FUP_feature_name] = 1

        for patient in self.all_patients:
            correct_FUPPREDICTION_New_Columns(patient_object=patient, FUP_feature_name="diabmelfu_PWELLS", BASELINE_feature_name="diabmel")
            correct_FUPPREDICTION_New_Columns(patient_object=patient, FUP_feature_name="hypertfu_PWELLS", BASELINE_feature_name="hyprthx")
            correct_FUPPREDICTION_New_Columns(patient_object=patient, FUP_feature_name="atrfibfu_PWELLS", BASELINE_feature_name="atrfib")
            
    def fill_missing_genotype_data(self):
        """For the patients without genotype data, the mode of the genotype will be assigned to them.
        """
        #Concatenate all the genotype data so that we can calculate the mode
        genotype_data = []
        for patient in self:
            if patient.GENOTYPE is not None:
                genotype_columns = patient.GENOTYPE.columns
                genotype_data.append(patient.GENOTYPE.copy().to_numpy().flatten())

        #Dataframe containing all the available genotype data
        genotype_df = pd.DataFrame(genotype_data, columns=genotype_columns)

        #Mode of the genotype data to be used for the patients with missing genotype
        genotype_mode = stats.mode(genotype_df.drop(["uniqid"] ,axis=1).to_numpy(), keepdims=False).mode

        #For patients without genotype data, use the mode genotype
        #A column is added to the genotype data indicating if the genotype data is missing or not
        for patient in self:
            if patient.GENOTYPE is None:
                filled_genotype_data = np.insert(genotype_mode, 0, patient.uniqid, axis=0)
                filled_genotype_data = np.append(filled_genotype_data, [1], axis=0)
                patient.GENOTYPE = pd.DataFrame(filled_genotype_data.reshape(1, len(filled_genotype_data)), columns= list(genotype_df.columns)+["missing-genoype"])        
            else:
                filled_genotype_data = patient.GENOTYPE.to_numpy().flatten()
                filled_genotype_data = np.append(filled_genotype_data, [0], axis=0)
                patient.GENOTYPE = pd.DataFrame(filled_genotype_data.reshape(1, len(filled_genotype_data)), columns= list(genotype_df.columns)+["missing-genoype"])

    def get_genotype_data(self):
        if not self.missing_genotypes_filled:
            print("Filling the missing genotype data values first (with the modes).")
            self.fill_missing_genotype_data()
            self.missing_genotypes_filled = True
        else:
            print("The missing genotype values are already replaced with the mode.")
        
        
        genotype_data = []
        for patient in self:
            genotype_data.append(patient.GENOTYPE.copy().to_numpy().flatten())

        #Dataframe containing all the available genotype data
        return pd.DataFrame(genotype_data, columns=patient.GENOTYPE.columns).set_index("uniqid")
                