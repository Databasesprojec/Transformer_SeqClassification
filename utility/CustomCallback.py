from transformers import TrainerCallback
import copy

class CustomCallback(TrainerCallback):
    """
    CustomCallback

    Helps us to track and gather log histories of individual trials during hyperparameter search.
        
        Args:
            trainer (Trainer): Instance of class Trainer (CustomTrainer).

        Methods:
            on_train_end(None): Event called at the end of training. Saves evaluation log history after each full training run.

    """
    def __init__(self, trainer):
        self._trainer = trainer
        self.all_log_history = []

    def on_train_end(self, args, state, control, **kwargs):
        """
        Event called at the end of training. Saves evaluation log history after each full training run in class attribute.

        Args:
            args (TrainingArguments): A class containing all training arguments used to control a Trainer's training process.
            state (TrainerState): A class containing a Trainer's inner state that will be saved along the model and optimizer when checkpointing
            control (TrainerControl): A class that handles a Trainer's control flow. Used by callbacks to activate some switches in training loop

        Returns:
            None
        """
        # Save a copy of the log history at the end of each training
        self.all_log_history.append(copy.deepcopy(self._trainer.state.log_history))