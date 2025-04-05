#!/usr/bin/env python
# coding: utf-8

# # Package Imports

# In[27]:


import numpy as np
import pandas as pd
import os
import re


from datasets import (
     load_from_disk,
     DatasetDict,
     load_dataset
)
import evaluate
from transformers import (
     AutoTokenizer,
     DataCollatorWithPadding,
     TrainingArguments,
     AutoModelForSequenceClassification,
     Trainer,
     logging,
     get_scheduler,

)
import torch
from ray import tune, train
import pickle
import optuna
from datetime import datetime
import utility.utility as util
import utility.CustomTrainer as ct
import utility.ModelConfig as mc
import utility.CustomCallback as cb


# # Global Settings:

# In[28]:


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
Save checkpoints strategy during training runs. Checkpoints are needed to resume training from specific stages.
Checkpoints require a lot of disk space. Turn off if disk space limit.
-"no": no checkpoints saved
-"epoch": checkpoint saved after every epoch
"""
_save_strategy = "no"

"""
Number of trials to run during this run of hyperparameter search.
"""
_num_trials = 2 


# # First run or continuation of hyperparameter search (HPS)

# In[29]:


"""
Boolean flag to indicate first run or continuation of HPS.
-"True": First run
-"False": Continuation - set _name_config_file below!
"""
_flag_first_run = True

"""
Set name of ModelConfig file for continuation of HPS.
"""
_name_config_file = ""

"""
Filepath to ModelConfig file.
"""
path_file_modelconfig = os.path.join("modelconfigs", _name_config_file)

"""
Load ModelConfig for continuation of HPS.
"""
model_config = None
if not _flag_first_run:
    with open(os.path.join(path_cwd, path_file_modelconfig), "rb") as f:
        model_config = pickle.load(f)


# # Configure model behavior (first run)

# In[30]:


"""
Description of downstream classification task.
"""
_task = "Binary Classification _ with study object and hps log history"

"""
Pretrained transformer base model to be used during finetuning on downstream task.
This has to be picked from the pre-trained models on HuggingFace
in order to be compatible with the Trainer API.
"""
_base_model = "roberta-base"

"""
Boolean flag to reset classification head.
Ff _base_model has already been finetuned on a prio classificaion task,
we need to reset its classification head to allow for new task.
-"True": reset model head
-"False": don't reset model head
"""
_reset_model_head = False

"""
Select loss function.
Three custom loss functions have been implemented with utility.CustomTrainer:
  f1: soft-f1 score
  mcc: soft-mcc
  wce: weighted cross entropy
  ce: standard cross entropy
"""
_loss_fct = "ce"

"""
Weighting scheme, only relevant when weighted-cross-entropy or other weighted
loss schemes are used.
"""
_weight_scheme = "rev_prop"

"""
Set evaluation metrics to be listed during training/evaluation:
"""
_eval_metrics = ["accuracy", "precision", "recall", "f1", "matthews_correlation"]


"""
Specify which metric should be maximized/minimized during hyperparameter-search
- "eval_matthews_correlation": MCC
- "eval_f1": F1
- "eval_loss": Cross-Entropy
- any other metric passed to the compute_metrics function

Note also specify direction of optimization: "maximize"/"minimize"
"""
_metric_best_model = "eval_matthews_correlation"
_metric_direction = "maximize"

"""
Employ freezing of layers, options:
"unfrozen": all layers unfrozen
"frozen": some transformer layers frozen
"""
_frozen = "unfrozen"

"""
Location of training dataset.
"True": HuggingFace Hub
"False": local directory
"""
_from_hub = True

"""
Name of dataset on HuggingFace Hub.
"""
_dataset_name_hub = ""

"""
Name of directory that contains the local dataset.
"""
_dataset_name_local = ""

"""
Boolean flag to indicate majority voting/multi-segment approach
"""
_flag_mv = True

"""
Name of HPS study
"""
_study_name = "test"


# ## Define hyperparameter search space

# In[31]:


"""
Adjust hyperparameters and their ranges as desired

https://huggingface.co/docs/transformers/en/hpo_train
"""
# Define hp space function
def optuna_hp_space(trial):
  return  {"learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
           "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
           "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1),
           "warmup_ratio": trial.suggest_float("warmup_ratio", 0, 1e-1),}


# # Configure using ModelConfig (continuation of HPS)

# In[32]:


"""
If continuation of HPS study, config settings loaded from ModelConfig.
"""
if not _flag_first_run:
    _task = model_config.task
    _base_model = model_config.base_model
    _reset_model_head = model_config.reset_model_head
    _loss_fct = model_config.loss_fct
    _weight_scheme = model_config.weight_scheme
    _eval_metrics = model_config.eval_metrics
    _metric_best_model = model_config.metric_best_model
    _metric_direction = model_config.metric_direction
    _frozen = model_config.frozen
    _from_hub = model_config.from_hub
    _dataset_name_hub = model_config.dataset_name_hub
    _dataset_name_local = model_config.dataset_name_local
    _flag_mv = model_config.flag_mv
    _study_name = model_config.study_name


# # Metadata

# In[33]:


"""
Timestamp of initial training. Used for file and directory naming schemes
"""
timestamp = datetime.now().strftime("%d_%m_%y_%H_%M")
if not _flag_first_run:
    timestamp = model_config.timestamp_initial

"""
Some model names contain '/' characters which create issues with file and directory pathing.
We replace them with '__' only for naming purposes
"""
base_model_altered = re.sub(r'/', '___', _base_model)

"""
Name of dataset, used to name model_config. Also replaces "/" with "_".
"""
dataset_name = re.sub(r'/', '_',_dataset_name_hub) if _from_hub else _dataset_name_local

"""
Directory path for training data.
"""
path_initial_training =  os.path.join("training_data" , base_model_altered, "initial_training" + "_" + timestamp)

"""
Select weighting scheme, only needed when using weighted cost functions.
Functions can be found in utility.utility.py
"""
class_weighting_schemes = {"rev_prop": util.get_reverse_prop_class_weights}

"""
Path to folder with local dataset.
"""
path_dataset_local = os.path.join("datasets" , _dataset_name_local)

"""
Name and path to ModelConfig object file.
"""
file_modelconfig = "ModelConfig_" + base_model_altered + "_" + dataset_name + "_" + timestamp + ".pkl"
path_file_modelconfig = os.path.join("modelconfigs", file_modelconfig)

"""
Path to sqlite-database with optuna HPS-study data.
"""
path_study_db = os.path.join("study_dbs", _study_name + "_" + base_model_altered + "_" + dataset_name + "_" + timestamp + ".db")


# # Setup

# ## Load Data

# In[34]:


raw_datasets = util.load_data(_from_hub, _dataset_name_hub, os.path.join(path_cwd, path_dataset_local))


# ## Determine number of labels/classes

# In[35]:


num_labels = util.get_no_labels(raw_datasets)


# ## Determine class weights

# In[36]:


class_weights = class_weighting_schemes[_weight_scheme](raw_datasets)
if not _flag_first_run:
    class_weights = model_config.class_weights


# ## Load Tokenizer

# In[37]:


tokenizer = AutoTokenizer.from_pretrained(_base_model)


# ## Function that returns the Tokenizer - needed to employ data mapping.
# 
# Note: Adjust this to desired task.

# In[38]:


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


# ## Tokenize dataset

# In[39]:


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


# ## Instantiate DataCollator
# Note: DataCollatorWithPadding allows for dynamic padding for individual batches. Only use with GPUs. For TPUs, use max_length padding attribute with Tokenizer instance.

# In[40]:


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ## Create instance of TrainingArguments
# 
# Adjust to desired behaviour. Most arguments learned during hyperparameter-search.

# In[41]:


"""
Create instance of class TrainingArguments.
"""
training_args = TrainingArguments(
    output_dir = os.path.join(path_cwd, path_initial_training),
    save_strategy = _save_strategy,
    eval_strategy = "epoch",
    logging_strategy = "epoch",
    metric_for_best_model = _metric_best_model,
    disable_tqdm = _disable_tqdm,
    )


# ## Model Initialzation

# In[42]:


"""
Here we supply two model init functions, one that freezes a number of encoder layers and
one that leaves all unfrozen.

Pass desired init function to Trainer below.

Gradual unfreezing helps to strike a balance between leveraging pre-trained
knowledge and adapting to task-specific data. By unfreezing layers gradually
during training, the model learns to prioritize retaining general linguistic
knowledge in the early layers while fine-tuning the higher layers to adapt to
task-specific nuances. This mitigates overfitting by allowing the model to
gradually specialize on the new task without abruptly forgetting the
linguistic representations learned during pre-training, resulting in more
effective adaptation and improved generalization to the target task.

Note: When utilizing gradual unfreezing you will have to train the model in
multiple steps. Gradually unfreezing ever more layers during training.
You will observe slower convergence, as such this will take more time.

Note: Depending on the choice of a base model and the desired number of layers
to freeze the model_init_frozen function might have to be adjusted.
To see which layers are available run:

  for name, param in model.named_parameters():
    print(name, param)

Observe entire model architecture and note layers you wish to freeze. Adjust
*conditional statement accordingly.

# https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751
"""


def model_init_frozen(freeze_layers):
  model = AutoModelForSequenceClassification.from_pretrained(_base_model, num_labels=num_labels, return_dict=True, ignore_mismatched_sizes=_reset_model_head)
  for name, param in model.named_parameters():
    # *conditional statement: currently all encoder layers are frozen
    freeze_layers = ["layer." + str(i) for i in range(11)]
    for fl in freeze_layers:
      if fl in name:
        param.requires_grad = False
  return model

def model_init():
  return AutoModelForSequenceClassification.from_pretrained(_base_model, num_labels=num_labels, return_dict=True, ignore_mismatched_sizes = _reset_model_head)


model_inits = {"unfrozen": model_init, "frozen": model_init_frozen}


# ## Create evaluation metric object and compute function to pass to Trainer.

# In[43]:


clf_metrics = evaluate.combine(_eval_metrics)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions = predictions, references = labels)


# ## Initialize CustomTrainer

# In[44]:


trainer = ct.CustomTrainer(
    type_loss = _loss_fct,
    model_init = model_inits[_frozen],
    class_weights = class_weights,
    args = training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics = compute_metrics,
)


# ## Create and add CustomCallback to Trainer. Allows us to save training logs after each hyperparameter trial run.

# In[45]:


callback = cb.CustomCallback(trainer)
trainer.add_callback(callback)


# ## (Optional) Create and assign an Optimizer and Scheduler
# 
# When using the HuggingFace Trainer API for hyperparameter search, we can no longer use the "optimizer" argument directly. Instead we need to create instances of both the optimizer and scheduler class, and then pass both to Trainer object.
# 
# Note: This is optional, as we could skip the following step and use the defaults. Included in case some custom behaviour is desired. Remove "#" to uncomment passing of objects.

# In[46]:


optimizer = torch.optim.AdamW(trainer.model.parameters())
lr_scheduler = get_scheduler(
    "linear",
    optimizer = optimizer,
    num_warmup_steps = 0,
    num_training_steps = training_args.num_train_epochs * tokenized_datasets["train"].num_rows

)

"""
Pass instances to Trainer
"""
#trainer.optimizers = (optimizer, lr_scheduler)


# # Hyperparameter Search via Optuna
# 
# https://towardsdatascience.com/state-of-the-art-machine-learning-hyperparameter-optimization-with-optuna-a315d8564de1
# 
# https://huggingface.co/docs/transformers/hpo_train
# 
# https://github.com/bayesian-optimization/BayesianOptimization
# 
# 

# In[47]:


# Define objective function that later selects best model based upon specific metric
def compute_objective(metrics):
  return metrics[_metric_best_model]


# ## Run Hyperparameter Search

# In[48]:


# Run hyperparameter search
best_run = trainer.hyperparameter_search(
    direction=_metric_direction,
    backend="optuna",
    hp_space = optuna_hp_space,
    n_trials = _num_trials,
    compute_objective = compute_objective,
    study_name=_study_name,
    storage= "sqlite:///" + os.path.join(path_cwd, path_study_db),
    load_if_exists=True,
    )
best_run


# ## Process HPS log history

# In[49]:


hps_log_df = util.process_hps_log_history(callback.all_log_history)


# # Create ModelConfig File

# In[50]:


if _flag_first_run:
    model_config = mc.ModelConfig(timestamp = timestamp,
                              base_model = _base_model,
                              reset_model_head = _reset_model_head,
                              task = _task,
                              loss_fct = _loss_fct,

                              from_hub = _from_hub,
                              dataset_name_hub = _dataset_name_hub,
                              dataset_name_local = _dataset_name_local,
                              path_dataset_local = path_dataset_local,

                              num_labels = num_labels,
                              weight_scheme = _weight_scheme,
                              class_weights = class_weights,
                              eval_metrics = _eval_metrics,
                              metric_best_model = _metric_best_model,
                              metric_direction = _metric_direction,

                              num_trials = _num_trials,
                              frozen = _frozen,
                              path_initial_training = path_initial_training,
                              best_run = best_run,
                              hps_log_df = hps_log_df,
                              flag_mv = _flag_mv,
                              study_name = _study_name,
                              path_study_db = path_study_db,)
else:
    # update model_config
    model_config.no_trials = model_config.num_trials + _num_trials
    model_config.best_run = best_run
    model_config.hps_log_df = util.merge_hps_log_histories(model_config.hps_log_df, hps_log_df)


# # Save ModelConfig

# In[51]:


with open(os.path.join(path_cwd, path_file_modelconfig), 'wb') as f:
    pickle.dump(model_config, f)

