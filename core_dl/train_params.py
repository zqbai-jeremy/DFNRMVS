import json


class TrainParameters:
    """ Store the parameters that used in training """

    """ Device ID """
    DEV_ID = 1

    """ Verbose mode (Debug) """
    VERBOSE_MODE = False

    """ Max epochs when training """
    MAX_EPOCHS = 4

    """ Mini-Batch Size """
    BATCH_SIZE = 6

    """ Learning Rate """
    START_LR = 1e-4

    """ Learning rate decay """
    LR_DECAY_FACTOR = 0.5

    """ Learning rate decay steps """
    LR_DECAY_STEPS = 1          # epochs

    """ Pytorch Dataloader worker threads number """
    TORCH_DATALOADER_NUM = 4    # threads

    """ Logging Steps """
    LOG_STEPS = 5               # per training-iterations

    """ Validation Steps """
    VALID_STEPS = 200           # per training-iterations

    """ Visualization Steps """
    VIS_STEPS = 100             # per training-iterations

    """ Validation Maximum Batch Number """
    MAX_VALID_BATCHES_NUM = 50

    """ Checkpoint Steps (iteration) """
    CHECKPOINT_STEPS = 5000      # per training-iterations

    """ Continue Step (iteration) """
    LOG_CONTINUE_STEP = 0            # continue step, used for logger

    """ Continue from (Dir) """
    LOG_CONTINUE_DIR = ''           # continue dir

    """ Description """
    DESCRIPTION = ''                # train description, used for logging changes

    def __init__(self, from_json_file=None):
        if from_json_file is not None:
            with open(from_json_file) as json_data:
                params = json.loads(json_data.read())
                json_data.close()

                # Extract parameters
                self.DEV_ID = int(params['dev_id'])
                self.MAX_EPOCHS = int(params['max_epochs'])
                self.BATCH_SIZE = int(params['batch_size'])
                self.START_LR = float(params['start_learning_rate'])
                self.LR_DECAY_FACTOR = float(params['lr_decay_factor'])
                self.LR_DECAY_STEPS = int(params['lr_decay_epoch_size'])
                self.TORCH_DATALOADER_NUM = int(params['loader_threads_num'])
                self.VERBOSE_MODE = bool(params['verbose'])
                self.VALID_STEPS = int(params["valid_per_batches"])
                self.MAX_VALID_BATCHES_NUM = int(params["valid_max_batch_num"])
                self.CHECKPOINT_STEPS = int(params['checkpoint_per_iterations'])
                self.VIS_STEPS = int(params['visualize_per_iterations'])
                self.LOG_CONTINUE_DIR = str(params['log_continue_dir'])
                self.LOG_CONTINUE_STEP = int(params['log_continue_step'])
                self.DESCRIPTION = str(params['description'])

    def save(self, json_file_path):
        params = dict()

        params['dev_id'] = self.DEV_ID
        params['max_epochs'] = self.MAX_EPOCHS
        params['batch_size'] = self.BATCH_SIZE
        params['start_learning_rate'] = self.START_LR
        params['lr_decay_factor'] = self.LR_DECAY_FACTOR
        params['lr_decay_epoch_size'] = self.LR_DECAY_STEPS
        params['loader_threads_num'] = self.TORCH_DATALOADER_NUM
        params['verbose'] = self.VERBOSE_MODE
        params['valid_per_batches'] = self.VALID_STEPS
        params['valid_max_batch_num'] = self.MAX_VALID_BATCHES_NUM
        params['checkpoint_per_iterations'] = self.CHECKPOINT_STEPS
        params['visualize_per_iterations'] = self.VIS_STEPS
        params['log_continue_step'] = self.LOG_CONTINUE_STEP
        params['description'] = self.DESCRIPTION
        params['log_continue_dir'] = self.LOG_CONTINUE_DIR

        with open(json_file_path, 'w') as out_json_file:
            json.dump(params, out_json_file, indent=2)

    def report(self):
        params = dict()
        params['dev_id'] = self.DEV_ID
        params['max_epochs'] = self.MAX_EPOCHS
        params['batch_size'] = self.BATCH_SIZE
        params['start_learning_rate'] = self.START_LR
        params['lr_decay_factor'] = self.LR_DECAY_FACTOR
        params['lr_decay_epoch_size'] = self.LR_DECAY_STEPS
        params['loader_threads_num'] = self.TORCH_DATALOADER_NUM
        params['verbose'] = self.VERBOSE_MODE
        params['valid_per_batches'] = self.VALID_STEPS
        params['valid_max_batch_num'] = self.MAX_VALID_BATCHES_NUM
        params['checkpoint_per_iterations'] = self.CHECKPOINT_STEPS
        params['visualize_per_iterations'] = self.VIS_STEPS
        params['log_continue_step'] = self.LOG_CONTINUE_STEP
        params['description'] = self.DESCRIPTION
        params['log_continue_dir'] = self.LOG_CONTINUE_DIR

        for param_key in params.keys():
            print(param_key, ': ', str(params[param_key]))
