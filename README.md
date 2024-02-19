# Transformer_SeqClassification

This repo provides the code and the dataset used for the project "On the Pitfalls of Insufficient Disclosure of Commercial Databases’ Coding Policies – A Case Replication of Noncompliance with Mandatory IFRS Adoption".

## Description (H)

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

* In this project, we leveraged the power of Python and the Hugging Face Transformers library to fine-tune multiple encoder-only transformer models, including BERT and RoBERTa, on a sequence classification task. The Hugging Face Trainer module facilitated efficient training and fine-tuning processes, streamlining the experimentation with different model architectures. To replicate our environment and ensure reproducibility, a comprehensive list of the required packages, including dependencies and their versions, is provided in the requirements.txt file. These dependencies encompass fundamental libraries and Hugging Face Transformers, creating a seamless and consistent development environment for anyone wishing to explore to this project. ([requirements.txt](https://github.com/ilias-m-n/Transformer_SeqClassification_HFAPI/blob/main/requirements.txt))
* Additionally, it's important to note that certain packages in the requirements.txt file are OS-dependent, requiring manual installation based on the operating system you intend to run the code on.
* To install the correct version of [PyTorch](https://pytorch.org) please follow the installation guide provided via the link.

### Project Structure

The project repository is organized into distinct folders, each serving a specific purpose:

* datasets: This folder contains the local dataset used for both fine-tuning and testing trained models when datasets are not directly loaded from the HuggingFace Hub.
* modelconfigs: In this folder, you'll find serialized objects of the class ModelConfig, which store all the parameters defined during the initial setup of hyperparameter search and evaluation results observed during final training.
* prediction_results: Here you will find Excel files containing the results obtained by running Transformer_SeqClass_Inference.ipynb to classify unlabeled/labeled samples for inference.
* study_dbs: This folder contains SQLite databases that store information on studies from Optuna hyperparameter searches. These databases are crucial for continuing hyperparameter searches after the initial run.
* trained_model: After completing the training process with the previously identified best hyperparameters, this folder stores the trained model data used to later load model weights for inference and testing.
* training_data: This folder encompasses the training data and checkpoints obtained from running HuggingFace's Trainer train() or hyperparametersearch() functions.
* utility: This directory contains several Python files:
  * CustomCallback.py: A Python class inheriting from TrainerCallback, enabling the tracking and saving of log history after each trial run during hyperparameter search.
  * CustomTrainer.py: A Python class inheriting from Trainer, providing the capability to use custom loss functions during model fine-tuning.
  * ModelConfig.py: A Python class that records and saves all configurations and relative paths set during the initial setup of hyperparameter search, along with all evaluation results and relevant paths determined or set after final training.
  * utility.py: This file contains various helper functions utilized across the project.
* ModelConfig_Reader.ipynb: This Jupyter Notebook facilitates the reading of serialized objects of the class ModelConfig.
* Transformer_SeqClass_HyperParamSearch.ipynb: In this Jupyter Notebook, the initial hyperparameter search is performed, along with any subsequent searches.
* Transformer_SeqClass_FinalTraining.ipynb: This Jupyter Notebook is responsible for executing the final training process and saving the associated evaluation results and trained model weights.
* Transformer_SeqClass_Inference.ipynb: Lastly, this Jupyter Notebook allows you to generate predictions using a trained model derived from the execution of Transformer_SeqClass_FinalTraining.ipynb.

### Executing program

  To execute our project successfully, follow these steps:
  
  **1) Initial Hyperparameter Search (Transformer_SeqClass_HyperParamSearch.ipynb):**
  
  In this notebook, configure the settings based on the specific task at hand.
  
  - **Global Settings:**
    - `path_cwd`: Absolute file path to the project folder. Set manually only if the code is not directly executed from the project's root folder; otherwise, the path is determined automatically.
    - `_num_trials`: Indicate the number of trials to run during hyperparameter search.
  
  - **First run or continuation of hyperparameter search (HPS):**
    - Set `_flag_first_run` to TRUE and ignore all other settings.
  
  - **Configure model behavior (first run):**
    - Set variables to indicate what base_model to use, which loss function to optimize, which metric to optimize during hyperparameter search, and several more.
  
  - **Define hyperparameter search space:**
    - Specify the hyperparameters for trials, including their ranges and types.
  
  After configuring the settings, execute the code. This produces several outcomes: training data and checkpoints for each trial run, the best hyperparameters discovered, and an instance of the ModelConfig.py class.
  
  **2) (Optional) Continuation of Hyperparameter Search (Transformer_SeqClass_HyperParamSearch.ipynb):**
  
  Following the initial hyperparameter search and the creation of a ModelConfig.py instance, subsequent runs are easily performed by setting `_flag_first_run` to False and providing the name of the ModelConfig.py instance, created during the initial run, to `_name_config_file`. Additionally, adjust `_num_trials` to the desired number of trials to perform.
  
  The remaining settings are extracted from the provided ModelConfig.py instance. After notebook execution, the ModelConfig.py instance is updated.
  
  **3) Final Training (Transformer_SeqClass_FinalTraining.ipynb):**
  
  The final training requires minimal configuration as most crucial variables are extracted from the ModelConfig.py instance.
  
  - Set `path_cwd` if the code is not directly executed from the project's root folder; otherwise, the path is determined automatically.
  - Provide the name of the ModelConfig.py instance produced during the initial hyperparameter search.
  - Execute the notebook.
  
  This step yields evaluation results on the test set, saves the trained model under `trained_models`, and updates the ModelConfig.py instance. Ensure that dependencies listed in `requirements.txt` are installed, and refer to the code documentation for detailed information on each variable. Note that all variables that have to be set manually, with exception of path_cwd, have an underscore prefix (_variable_name), while those set by the code have no underscore prefix (variable_name).

## Data

### Datasets (H)

Description of datasets used to train models. Their make up and locations/names of Hub.

[HF-Hub](https://huggingface.co/datasets/Databasesprojec)

### Origin of data (H)

How was the data complied, extracted, cleaned, and any other steps applied.

### Dataset preparation (for model training) (H)

Which part of the data are we using for training and why. (H)

## Appendix (H,S)

You can find the online appendix for this project [here](https://github.com/Databasesprojec/Transformer_SeqClassification/blob/main/Online%20Appendix.xlsx). Please be advised that this file is not viewable directly on GitHub. Kindly download it to access its contents.
