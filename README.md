# Transformer_SeqClassification

This repo provides the code used for the project "Following the blind? Database Policies and the Case of IFRS Noncompliance".

You find the datasets we used on Hugging Face Hub. [here](https://huggingface.co/Databasesprojec)


## Description Project

We present a case illustrating the pitfalls of insufficient disclosure of commercial databases’ coding policies. In response to the identified data gap, we showcase the application of Encoder-Only Transformer models (e.g., BERT, RoBERTa, FinBERT) to extract consolidation status and offer guidance for coding IFRS-mandated firms. This GitHub repository provides access to the customizable Python code used to fine-tune and train our Encoder-Only models. Additionally, on HuggingFace we offer researchers our training/validation/test and prediction datasets, the "BERT base" model’s predicted classifications, and an identifier file with links to Worldscope. The training datasets are described below.

## Getting Started

### Dependencies

This project is built using Python 3.12.8 and leverages the Hugging Face Transformers library to fine-tune encoder-only transformer models—such as BERT and RoBERTa—for sequence classification tasks. The Hugging Face Trainer module simplifies the training and fine-tuning process, enabling efficient experimentation with different model architectures.

Important: Due to some version pinning issues with PyTorch on Python 3.12, please follow these steps to get the code running:

1) Manually install PyTorch Build (Stable 2.6.0):
Visit the [PyTorch](https://pytorch.org/get-started/locally/) website and install the appropriate build for your system.

2) Install remaining dependencies:
Once PyTorch is installed, proceed to install the rest of the required packages from the provided [requirements.txt](https://github.com/Databasesprojec/Transformer_SeqClassification/blob/main/requirements.txt) file.

Note: The requirements.txt file was created on a Linux distribution and includes some OS-dependent packages. You may need to adjust or manually install certain packages if you are running on a different operating system.

### Code Structure

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

 This repository offers both Jupyter notebooks and traditional Python scripts to cater to your workflow preferences. You can use the notebooks for an interactive environment or opt for the Python   scripts if you should so prefer.

  To execute our project successfully, follow these steps:
  
  **1) Initial Hyperparameter Search (Transformer_SeqClass_HyperParamSearch):**
  
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
  
  **2) (Optional) Continuation of Hyperparameter Search (Transformer_SeqClass_HyperParamSearch):**
  
  Following the initial hyperparameter search and the creation of a ModelConfig.py instance, subsequent runs are easily performed by setting `_flag_first_run` to False and providing the name of the ModelConfig.py instance, created during the initial run, to `_name_config_file`. Additionally, adjust `_num_trials` to the desired number of trials to perform.
  
  The remaining settings are extracted from the provided ModelConfig.py instance. After notebook execution, the ModelConfig.py instance is updated.
  
  **3) Final Training (Transformer_SeqClass_FinalTraining):**
  
  The final training requires minimal configuration as most crucial variables are extracted from the ModelConfig.py instance.
  
  - Set `path_cwd` if the code is not directly executed from the project's root folder; otherwise, the path is determined automatically.
  - Provide the name of the ModelConfig.py instance produced during the initial hyperparameter search.
  - Execute the notebook.
  
  This step yields evaluation results on the test set, saves the trained model under `trained_models`, and updates the ModelConfig.py instance. Ensure that dependencies listed in `requirements.txt` are installed, and refer to the code documentation for detailed information on each variable. Note that all variables that have to be set manually, with exception of path_cwd, have an underscore prefix (_variable_name), while those set by the code have no underscore prefix (variable_name).

## Data

### Datasets

All of our datasets used in training or for prediction can be acccessed via HuggingFace
on this hub you can find:
1. identifiers: file contains unique identifier for each report downloaded from Perfect Information and used in the training or prediction process. in identifiers we provide:
   * filename: unique identifer for each report downloaded from Perfect Information database.
   * wc06035: Worldscope identifier.
   * Year: year as defined by PW.
3. Our training and prediction datasets, for each dataset, we provide the following variables:
    * label: binary consolidation label for the report (1=consolidated, 0=unconsolidated)
    * id: unique filename identifier
    * text: text extract from the report that was used in the training process
   We provide the following training datasets:
  * English_short_window_training: dataset used for training for English language reports using our short window (one segment of 512 tokens) specification.
  * English_long_window_training: dataset used for training for English language reports using our short window (six segments of 512 tokens) specification.
  * German_training: dataset used for training for German language reports using our short window (one segment of 512 tokens) specification.
  * French_training: dataset used for training for French language reports using our short window (one segment of 512 tokens) specification.
  and we provide the following prediction datasets:
  * English_short_window_Predict1 to English_short_window_Predict5: Predictions for the English report sample using our short window specification.
  * English_Long_window_Predict: Predictions for the English report sample using our long-window specification.

### Origin of Data

Our training and prediction datasets are obtained from Perfect Information database as collected by Daske et al. (2023).
* For the English language training sample: We pre-select a sample of 2,326 documents to include more unconsolidated reports in our dataset, and we manually label these reports (54% labeled as consolidated and 46% as unconsolidated)
* For the French language training sample: our sample of 5,000 documents (90% labeled as consolidated), is randomly extracted from our Perfect Information dataset, and we reply on the Worldscope data item “Accounting Method For Long Term Investment>50%” (WC07531) for labeling these reports since it is reliable for the years before 2012
* For the German language training sample: our sample of 7,000 documents (90% labeled as consolidated), is randomly extracted from our Perfect Information dataset, and we reply on the Worldscope data item “Accounting Method For Long Term Investment>50%” (WC07531) for labeling these reports since it is reliable for the years before 2012.

## Links

* [HuggingFace](https://huggingface.co/Databasesprojec)
* Daske, H., Sergeni, C., & Uckert, M. (2023). Unstandardized accounting language [Working paper].  
