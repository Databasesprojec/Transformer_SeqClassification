class ModelConfig:
    """
    Instances of class ModelConfig save the meta information, model configurations, and evaluation metrics
    set or produced during configuration and finetuning of Transformer models on downstream classification task.
    Initialization occurs after first hyperparameter search run. Respective attributes are updated after each following run.
    Evaluation metrics produced after final training and evaluation on test set are subsequently updated after running final training.

    Attributes:
    
        timestamp_initial (datetime.datetime): Date and time of initial hyperparameter search.
        
        base_model (String): Name of model on HuggingFace Hub used on downstream task. 
        
        reset_model_head (Boolean): Indicates whether classification head needs to be reset. Set to True when model already comes with finetuned
                                    classification head.
                                    
        task (String): Description of downstream task.
        
        loss_fct (String): Name of loss function to be used during model optimization/training.

        from_hub (Boolean): Flag indicating whether training dataset is loaded from the HuggingFace Hub (True) or a local directory (False).
        
        dataset_name_hub (String): Name of dataset on HuggingFace Hub.
        
        dataset_name_local (String): Name of local directory containing the local dataset to be used during training.
        
        path_dataset_local (String): Relative file path to local directory containing the local dataset.
        
        num_labels (Integer): Number of unique labels in dataset (Binary = 2, Multiclass > 2).
        
        weight_scheme (String): Weight scheme to be used when utilizing a weighted loss function scheme.

        class_weights (Dictionary): Dictionary containing the labels as keys and their weights as values. Only used when utilizing a weighted loss function
                                    scheme.

        metric_best_model (String): Metric to be maximized/minimized during hyperparameter search.

        metric_diretion (String): Indicates whether metric_best_model should be maximized (e.g., F1 or MCC) or minimized (e.g., cross-entropy loss).

        num_trials (Integer): Number of trial runs performed for hyperparameter search.

        frozen (Boolean): Flag indicating whether some layers have been frozen during training.

        path_initial_training (String): Relative filepath to training data (e.g., checkpoints) created during hyperparameter search.

        best_run (Dictionary): Contains information on the best trial run performed during hyperparameter search.

        hps_log_df (DataFrame): Contains log history of each trial run performed during hyperparameter search.

        flag_mv (boolean): Indicates whether a majority voting/sliding window scheme should be performed.

        study_name (String): Name of hyperparamete search study object.

        path_study_db (String): Relative path to database containing meta information on each trail run performed during hyperparameter search.

        time_stamp_final (datetime.datetime): Date and time of final training.

        path_final_training (String): Relative filepath to training data (e.g., checkpoints) created during final training.

        path_trained_model (String): Relative filepath to trained model data created during final training.

        training_log_df (DataFrame): Contains evaluation metrics for each epoch produced during final training.

        predictions_df (DataFrame): Contains the true and predicted labels for each sample in the test dataset

        evaluation_results (Dictionary): Contains evaluation metrics produced by testing model performance on test dataset.

        confusion_matrix (ndarray): Contains confusion matrix.
        
        predictions_mv_df (DataFrame): Contains the true and predicted labels for each aggregated sample in the test dataset

        evaluation_results_mv (Dictionary): Contains evaluation metrics produced by testing model performance on test dataset and aggregating predictions
                                            for each document's segments.

        confusion_matrix_mv (ndarray): Contains confusion matrix for aggregated predicted labels.
                                            
    """

    def __init__(self, 
                 timestamp, 
                 base_model,
                 reset_model_head,
                 task, 
                 loss_fct, 

                 from_hub,
                 dataset_name_hub,
                 dataset_name_local, 
                 path_dataset_local, 
                 
                 num_labels,
                 weight_scheme, 
                 class_weights,
                 eval_metrics,  
                 metric_best_model,
                 metric_direction,
                 
                 num_trials, 
                 frozen,
                 path_initial_training,
                 best_run,
                 hps_log_df,
                 flag_mv,
                 study_name,
                 path_study_db,):

        # set during initialization 
        self.timestamp_initial = timestamp
        self.base_model = base_model
        self.reset_model_head = reset_model_head
        self.task = task
        self.loss_fct = loss_fct

        self.from_hub = from_hub
        self.dataset_name_hub = dataset_name_hub
        self.dataset_name_local = dataset_name_local
        self.path_dataset_local = path_dataset_local


        
        self.num_labels = num_labels
        self.weight_scheme = weight_scheme
        self.class_weights = class_weights
        self.eval_metrics = eval_metrics 
        self.metric_best_model = metric_best_model
        self.metric_direction = metric_direction
        
        self.num_trials = num_trials
        self.frozen = frozen
        self.best_run = best_run
        self.path_initial_training = path_initial_training
        
        self.flag_mv = flag_mv

        # hyperparameter search
        self.hps_log_df = hps_log_df
        self.study_name = study_name
        self.path_study_db = path_study_db
        
        # set during final training after hyperparameter search
        self.timestamp_final = None
        self.path_final_training = None
        self.path_trained_model = None

        # set during performance evaluation
        self.training_log_df = None
        self.predictions_df = None
        self.evaluation_results = None
        self.confusion_matrix = None
        # majority voting
        self.predictions_mv_df = None
        self.evaluation_results_mv = None
        self.confusion_matrix_mv = None