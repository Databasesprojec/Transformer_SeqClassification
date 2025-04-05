#!/usr/bin/env python
# coding: utf-8

# # Package Imports

# In[ ]:


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
)
import torch
from ray import tune, train
import pickle
from datetime import datetime
from sklearn.metrics import confusion_matrix
import utility.utility as util
import utility.CustomTrainer as ct
import utility.ModelConfig as mc


# # Set ModelConfig

# In[ ]:


"""
Path to project root-directory, needs to be set if not directly called from prjoect directory.
"""
path_cwd = os.getcwd()

"""
Name of ModelConfig file
"""
_name_config_file = ""

"""
Filepath to ModelConfig
"""
path_file_modelconfig = os.path.join("modelconfigs", _name_config_file)

"""
Name of dataset on Hub to be used during inference.
"""
_name_dataset_hub = ""


# # Load ModelConfig

# In[ ]:


model_config = None
with open(os.path.join(path_cwd, path_file_modelconfig), "rb") as f:
    model_config = pickle.load(f)


# # Filepath to trained model

# In[ ]:


path_trained_model = os.path.join(path_cwd, model_config.path_trained_model)


# # Load Dataset

# In[ ]:


raw_dataset = util.load_data(True, _name_dataset_hub, "")["train"]


# # Load Tokenizer and tokenize dataset

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(path_trained_model)


# In[ ]:


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


# In[ ]:


tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)


# In[ ]:


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# # Evaluation Metrics

# In[ ]:


clf_metrics = evaluate.combine(model_config.eval_metrics)

def compute_metrics(eval_preds):
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return clf_metrics.compute(predictions = predictions, references = labels)


# # Model & Trainer Initialization

# In[ ]:


model = AutoModelForSequenceClassification.from_pretrained(path_trained_model)


# In[ ]:


trainer = Trainer(model,
    data_collator=data_collator,
    processing_class=tokenizer,)


# # Create predictions

# In[ ]:


predictions = trainer.predict(tokenized_dataset)


# # Process predictions and merge with dataset

# In[ ]:


pred_df, pred_mv_df = util.process_prediction_results(raw_dataset.to_pandas(), predictions, "original_id", "text", "id", "label", model_config.flag_mv)


# # Prepare meta dataframe

# In[ ]:


meta_dict = {"date": datetime.now().strftime("%d_%m_%y"),
             "time": datetime.now().strftime("%H_%M"),
             "base_model": model_config.base_model,
             "trained model path": model_config.path_trained_model,
             "dataset": _name_dataset_hub,
             "modelconfig": _name_config_file}
meta_df = pd.DataFrame.from_dict(meta_dict, columns=[""], orient="index")


# # Prepare column descriptions

# In[ ]:


col_descr_dict = {"label": "True label for sample.",
                  "id": "Sample ID, if segmented data this represents the segments ID.",
                  "original_id": "If segmented, this represents the original sample ID.",
                  "pred_logits": "Model's raw logit outputs.",
                  "pred_label": "Predicted label based on model's raw logit outputs",
                  "pred_mv_agg_logits": "If segmented data, raw logit outputs aggregated over each segment per original sample ID",
                  "pred_mv_agg_logits_label": "If segmented data, predicted label based on aggregated raw logits",
                  "pred_mv_agg_label": "If segmented data, predicted labels aggregated over each segment's predicted label per original sample ID"
                 }
col_descr_df = pd.DataFrame.from_dict(col_descr_dict, columns=[""], orient="index")


# # Save results

# In[ ]:


path_excel_file = os.path.join(path_cwd ,"prediction_results", "pred_" + datetime.now().strftime("%d_%m_%y_%H_%M") + ".xlsx")


# In[ ]:


with pd.ExcelWriter(path_excel_file) as writer:
    meta_df.to_excel(writer, sheet_name="Meta", index = True)
    pred_df.to_excel(writer, sheet_name="Predictions", index=False)
    if model_config.flag_mv:
        pred_mv_df.to_excel(writer, sheet_name="Predictions_MV", index = False)
    col_descr_df.to_excel(writer, sheet_name="Data Description", index = True)

