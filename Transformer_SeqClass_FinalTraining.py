#!/usr/bin/env python
# coding: utf-8

# # Package Imports

# In[2]:


import numpy as np
import pandas as pd
import os
import re


from datasets import load_from_disk, concatenate_datasets, DatasetDict, load_dataset
import evaluate
from transformers import (
     AutoTokenizer,
     DataCollatorWithPadding,
     TrainingArguments,
     AutoModelForSequenceClassification,
     Trainer,
     logging,
     get_scheduler,
     TrainerCallback,
)
import torch
from ray import tune, train
import pickle
from datetime import datetime
from sklearn.metrics import confusion_matrix
import utility.utility as util
import utility.CustomTrainer as ct
import utility.ModelConfig as mc

# resets import once changes have been applied
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Global Settings:

# In[3]:


"""
Filepath to project-root folder, needs to be set manually if not directly called from project folder.
"""
path_cwd = os.getcwd()

"""
Disables/enables progress bars during training of model.
-"True": no progress bars shown during training
-"False": progress bars shown during training
"""
_disable_tqdm = False

"""
Name of ModelConfig file created during HPS
"""
_name_config_file = ""


# # Load ModelConfig

# In[4]:


"""
path to file with modelconfig
"""
path_file_modelconfig = os.path.join("modelconfigs", _name_config_file)

model_config = None
with open(os.path.join(path_cwd, path_file_modelconfig), "rb") as f:
    model_config = pickle.load(f)


# # Add Metadata to ModelConfig

# In[5]:


"""
Timestamp of final training. Used for file and directory naming schemes.
"""
model_config.timestamp_final = datetime.now().strftime("%d_%m_%y_%H_%M")

"""
Alter base model name for naming of path
"""
base_model_altered = re.sub(r'/', '___', model_config.base_model)

"""
Directory path to final training data.
"""
model_config.path_final_training = os.path.join("training_data" , base_model_altered, "final_training" + "_" + model_config.timestamp_final)

"""
Weighting schemes.
"""
class_weighting_schemes = {"rev_prop": util.get_reverse_prop_class_weights}

"""
Path to folder with trained model
"""
model_config.path_trained_model = os.path.join("trained_models", base_model_altered + "_" + model_config.timestamp_final)


# # Setup

# ## Load Data

# In[6]:


raw_datasets = util.load_data(model_config.from_hub, model_config.dataset_name_hub, os.path.join(path_cwd, model_config.path_dataset_local))


# During final training we merge the training with valdidation dataset and train the model with the priorly found best hyperparameters.

# In[7]:


raw_datasets = util.prep_datasets_final_train(raw_datasets)


# ## Load Tokenizer

# In[8]:


tokenizer = AutoTokenizer.from_pretrained(model_config.base_model)


# ## Function that returns the Tokenizer - needed to employ data mapping.
# 
# Note: Adjust this to desired task.

# In[9]:


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


# ## Tokenize dataset

# In[10]:


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


# ## Instantiate DataCollator

# In[11]:


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ## Create instance of TrainingArguments
# 
# Initialize TrainingArguments with best_run hyperparameters found during HPS.

# In[12]:


training_args = TrainingArguments(
    output_dir = os.path.join(path_cwd, model_config.path_final_training),
    save_strategy = "epoch",
    eval_strategy = "epoch",
    logging_strategy = "epoch",
    disable_tqdm = _disable_tqdm,
    **model_config.best_run.hyperparameters,
    )


# ## Model Initialzation

# In[13]:


def model_init_frozen(freeze_layers):
  model = AutoModelForSequenceClassification.from_pretrained(model_config.base_model, num_labels=model_config.num_labels, return_dict=True, ignore_mismatched_sizes = model_config.reset_model_head)
  for name, param in model.named_parameters():
    # *conditional statement: currently all encoder layers are frozen
    freeze_layers = ["layer." + str(i) for i in range(11)]
    for fl in freeze_layers:
      if fl in name:
        param.requires_grad = False
  return model

def model_init():
  return AutoModelForSequenceClassification.from_pretrained(model_config.base_model, num_labels=model_config.num_labels, return_dict=True, ignore_mismatched_sizes = model_config.reset_model_head)

model_inits = {"unfrozen": model_init, "frozen": model_init_frozen}


# ## Create evaluation metric object and compute function to pass to Trainer.

# In[14]:


clf_metrics = evaluate.combine(model_config.eval_metrics)

def compute_metrics(eval_preds):
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return clf_metrics.compute(predictions = predictions, references = labels)


# # Initialize CustomTrainer

# In[15]:


trainer = ct.CustomTrainer(
    type_loss = model_config.loss_fct,
    model_init = model_inits[model_config.frozen],
    class_weights = model_config.class_weights,
    args = training_args,
    train_dataset=tokenized_datasets["train_val"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics = compute_metrics,
)


# # Train Model

# In[16]:


trainer.train()


# ## Save dataframe with evaluation metrics per training epoch

# In[17]:


model_config.training_log_df = util.process_log_history(trainer.state.log_history,
                                                        int(trainer.args.num_train_epochs))


# # Model evaluation on test data

# In[23]:


predictions = trainer.predict(tokenized_datasets["test"])


# # Process predictions

# In[24]:


raw_test_data_df = raw_datasets["test"].to_pandas()
pred_df , pred_mv_df = util.process_prediction_results(raw_test_data_df, predictions, "original_id", "text", "id", "label", model_config.flag_mv)
model_config.predictions_df = pred_df
model_config.predictions_mv_df = pred_mv_df


# ## Compute evaluation metrics and confusion matrix on test dataset

# In[25]:


model_config.evaluation_results = clf_metrics.compute(pred_df["label"], pred_df["pred_label"])
model_config.confusion_matrix = confusion_matrix(pred_df["label"], pred_df["pred_label"])


# ## If majority voting scheme employed do the same for aggragated predictions

# In[26]:


if model_config.flag_mv:
    model_config.evaluation_results_mv = clf_metrics.compute(pred_mv_df["label"], pred_mv_df["pred_mv_agg_logits_label"])
    model_config.confusion_matrix_mv = confusion_matrix(pred_mv_df["label"], pred_mv_df["pred_mv_agg_logits_label"])


# # Save Model

# In[27]:


trainer.save_model(os.path.join(path_cwd, model_config.path_trained_model))


# # Save Model Config

# In[28]:


with open(os.path.join(path_cwd, path_file_modelconfig), 'wb') as f:
    pickle.dump(model_config, f)

