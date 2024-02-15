from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from transformers import Trainer
import pandas as pd
import numpy as np
import json


def det_class_weights(datasets, subset = "train", label = "label"):
    """
    Determines the ratios of each class present in a Dataset.

    Args:
        datasets (DatasetDict): DatasetDict for which to determine class weights.
        subset (String): Name of Dataset in DatasetDict from which to take the 
                         class ratios to determine class weights.
        label (String): Column name of Dataset object that contains the class labels.

    Returns:
        Dictionary containing the fraction of each class in percent.
    """
    no_labels = Counter(datasets[subset][label])
    overall_no_samples = 0
    for l in no_labels:
        overall_no_samples += no_labels[l]
    weights = {l: no_labels[l]/overall_no_samples for l in no_labels}
    return weights

def get_reverse_prop_class_weights(datasets, subset = "train", label = "label", label_order = [0,1]):
    """
    Determines the inverse class weights to run with weighted cost functions.

    Args:
        datasets (DatasetDict): DatasetDict for which to determine class weights.
        subset (String): Name of Dataset in DatasetDict from which to take the 
                         class ratios to determine class weights.
        label (String): Column name of Dataset object that contains the class labels.
        label_order (List): List contains the order of labels. Needed to order the weights.

    Returns:
        Dictionary containing the inverse fractions of each class.
    """
    weights = det_class_weights(datasets, subset, label)
    rev_prop_weights = [1/weights[l] for l in label_order]
    return rev_prop_weights

def get_no_labels(datasets, subset = "train", label = "label"):
    """
    Determines the unique number of labels for a DatasetDict.

    Args:
        datasets (DatasetDict): DatasetDict for which to determine class weights.
        subset (String): Name of Dataset in DatasetDict from which to take determine
                         the unique class labels.
        label (String): Column name of Dataset object that contains the class labels.

    Returns:
        Integer indicating the number of unique labels.
    """
    labels = set(datasets[subset][label])
    return len(labels)

def load_data(from_hub = True, dataset_name_hub = "", path_dataset = ""):
    """
    Loads a DatasetDict from either a local directory or the HuggingFace Hub.

    Args:
        from_hub (Boolean): Determines whether to load dataset from Hub or a local directory.
        dataset_name_hub (String/List(String)): Name of Dataset to be loaded from Hub.
                                        If String: Corresponding dataset will be loaded.
                                        If List: Corresponding datasets will be loaded and merged.
        path_dataset (string): Filepath to local dataset.

    Returns:
        DatasetDict
    """
    if from_hub:
        if isinstance(dataset_name_hub, list):
            mult_datasets = []
            # load individual datasets
            for ds in dataset_name_hub:
                mult_datasets.append(load_dataset(ds))
            # concatenate datasets
            if len(mult_datasets) == 1:
                return mult_datasets[0]
            return merge_datasetdicts(mult_datasets)
        else:
            return load_dataset(dataset_name_hub)
    else:
        return load_from_disk(path_dataset)

def merge_datasetdicts(list_dsd):
    """
    Merges DatasetDicts along their splits.

    Args:
        list_dsd (List(DatasetDicts)): List containing DatasetDict objects.

    Returns:
        Merged DatasetDict.
    """
    datasets = {}

    for split in list_dsd[0]:
        datasets[split] = list_dsd[0][split]

    for i in range(1,len(list_dsd)):
        for split in datasets:
            datasets[split] = concatenate_datasets([datasets[split], list_dsd[i][split]])
    

    return DatasetDict(datasets)

def prep_datasets_final_train(raw_datasets, train = "train", val = "validation"):
    """
    Merges training and validation datasets for final training after validation.

    Args:
        raw_datasets (DatasetDict): DatasetDict containing the individual dataset splits.
        train (String): Name of Dataset in DatasetDict containing the training dataset.
        val (String): Name of Dataset in DatasetDict containing the validation dataset.

    Returns:
        DatasetDict where the training and validation datasets have been merged.
    """
    train_val = concatenate_datasets([raw_datasets[train], raw_datasets[val]])
    raw_datasets["train_val"] = train_val
    del raw_datasets["train"]
    del raw_datasets["validation"]
    return raw_datasets

def process_prediction_results(data, pred, id_col, seg_col, seg_id_col, label_col, flag_mv = False):
    """
    Aggregates prediction over individual segements when majority voting scheme was employed.

    Args:
        data (pd.DataFrame): DataFrame containing the original dataset.
        pred (String): Named tuple with following keys: 
                            predictions (np.ndarray): raw logits
                            label_ids (np.ndarray): labels (optional - only if dataset contained labels)
                            metrics (Dict[str, float]): dictionary of metrics (optional - only if dataset contained labels)
        id_col (String): Column name of ID column over which to aggregate individual logits before determining prediction label.
        seg_col (String): Column name of segment text column.
        seg_id_col (String): Column name of segement ID column.
        label_col (String): Column name of label column.
        flag_mv (Boolean): Indicate whether to aggregate results of segements per id
        

    Returns:
        Aggregated dataframe data containing the prediction labels for each ID in id_col.
        
    """
    res_df = data.copy()
    res_mv_df = None

    res_df["pred_logits"] = list(pred.predictions)
    res_df["pred_label"] = np.argmax(pred.predictions, axis=1)

    if flag_mv:
        res_mv_df = simple_majority_voting(res_df, id_col, seg_col, seg_id_col, label_col)

    res_df.drop([seg_col], axis=1, inplace=True)
        
    return res_df, res_mv_df

def simple_majority_voting(data, id_col, seg_col, seg_id_col, label_col):
    """
    Aggregates prediction over individual segements when majority voting scheme was employed.

    Args:
        data (pd.DataFrame): DataFrame containing the original dataset with unaggregated prediction labels and logits.
        id_col (String): Column name of ID column over which to aggregate individual logits before determining prediction label.
        seg_col (String): Column name of segment text column.
        seg_id_col (String): Column name of segement ID column.
        label_col (String): Column name of label column.

    Returns:
        Aggregated dataframe data containing the prediction labels for each ID in id_col.
        
    """
    res_mv_df = data.copy()

    # remove segment specific columns
    res_mv_df.drop([seg_col, seg_id_col], axis=1, inplace=True)
    # aggregate scores
    logits_agg = res_mv_df.groupby([id_col])["pred_logits"].apply(lambda x: np.mean(np.vstack(x), axis=0))
    pred_label_agg = res_mv_df.groupby([id_col])["pred_label"].mean()
    # rename series names
    logits_agg.rename("pred_mv_agg_logits", inplace = True)
    pred_label_agg.rename("pred_mv_agg_label", inplace = True)
    # remove segment specific logits and predicted labels
    res_mv_df.drop([col for col in res_mv_df.columns if col not in [id_col, label_col]], axis=1, inplace=True)
    # remove duplicate rows
    res_mv_df.drop_duplicates(inplace=True)
    # merge results back to original dataframe
    res_mv_df = pd.merge(res_mv_df, logits_agg, on = [id_col])
    res_mv_df = pd.merge(res_mv_df, pred_label_agg, on = [id_col])

    # determine label based on aggregated logits
    res_mv_df["pred_mv_agg_logits_label"] = res_mv_df["pred_mv_agg_logits"].apply(np.argmax)

    return res_mv_df

def process_log_history(log_hist, num_epoch):
    """
    Processes Trainer.log_history info into a dataframe after running Trainer.train()

    Args:
        log_hist (List): List containing the evaluation metrics for each epoch trained
        num_epoch (Integer): Number of training epochs performed 

    Returns:
        DataFrame containing the log information
    """

    train_logs = pd.DataFrame()
    eval_logs = pd.DataFrame()

    for i in range(num_epoch):
        train_logs = pd.concat([train_logs, pd.DataFrame(log_hist[i*2], index=[i+1])])
        eval_logs = pd.concat([eval_logs, pd.DataFrame(log_hist[1+i*2], index=[i+1])])

    # remove duplicate columns
    log = pd.concat([train_logs, eval_logs], axis=1)
    log = log.loc[:, ~log.columns.duplicated(keep='last')]

    return log

def process_hps_log_history(hps_log_hist):
    """
    Processes the individual log histories of each trial run performed during hyperparameter search

    Args:
        hps_log_hist (list): Contains the individual log histories of each trial

    Returns:
        DataFrame containing the log information for each run
    """

    res = pd.DataFrame()
    trial_no = 0
    
    for trial in hps_log_hist:
        num_epoch = int(trial[-1]["epoch"])
        int_hps_log = process_log_history(trial, num_epoch)
        int_hps_log["trial_no"] = trial_no
        trial_no += 1
        res = pd.concat([res, int_hps_log])

    res.reset_index(drop=True, inplace=True)
    return res

def merge_hps_log_histories(prev, new):
    """
    Merges two log history dataframe when multiple hyperparameter searches have been performed

    Args:
        prev (list): Contains the processed log history dataframe from previous runs
        new (list): Contains the processed log history dataframe from current runs

    Returns:
        DataFrame containing the log information for the combined runs
    """
    tmp_prev = prev.copy()
    tmp_new = new.copy()
    prev_no_trials = tmp_prev["trial_no"].values[-1]
    tmp_new["trial_no"] = tmp_new["trial_no"] + prev_no_trials
    res = pd.concat([tmp_prev, tmp_new])
    res.reset_index(drop=True, inplace=True)

    return res

def process_study_db_trial_params(tp):
    """
    Processes result from querying database (sqlite) object containing the hyperparameter choices for each trial run
    performed during hyperparameter search.

    SQL query performed: SELECT * FROM trial_params
    
    Args:
        tp (list): List containing query results

    Returns:
        DataFrame containing the log information for the combined runs
    """
    int_dict = {}
    
    for ele in tp:
        trial_no = ele[1]
        param_n = ele[2]
        param_v = ele[3]
        j = json.loads(ele[4])
        if j["name"] == "CategoricalDistribution":
            param_v = j["attributes"]["choices"][int(param_v)]
        int_list = int_dict.get(param_n, [])
        int_list.append(param_v)
        int_dict[param_n] = int_list       

    return pd.DataFrame(int_dict)
