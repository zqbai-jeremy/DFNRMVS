import datetime
import os
import socket
import warnings
import sys
import signal
import atexit
import torch
import json
from torch.autograd import Variable
from core_dl.logger_file import FileLogger
from core_dl.logger_tensorflow import TensorboardLogger


class Logger:
    """
    Logger for training,
    usage:
    # >>> logger = Logger("./runs", "csv|txt|tensorboard")
    # >>> logger.add_keys(['Loss/class_layer', 'Accuracy/class_layer'])
    # >>> logger.log({'Loss/class_layer': 0.04, 'Accuracy/class_layer': 0.4, 'Iteration': 4})
    Use prefix to classify the term:
    - The Scalars:'Loss' and 'Accuracy' 'Scalar'
    - The Scalars: 'Scalars' used for visualize multiple records in single figure, use 'dict' for value (Only in tensorboard)
    - The net instance: 'net' used for parameter histogram (Only in tensorboard)
    - The Image visualization: 'Image' used for visualize bitmap (Only in tensorboard)
    """

    """ Logger instance """
    loggers = {}

    """ The place where log files are stored """
    log_base_dir = ''

    def __init__(self, base_dir=None, log_types='csv|tensorboard', tag='', description='', ckpt_path=None, continue_from_step=None):
        '''
        :param base_dir: The base directory stores the log file
        :param log_types:  the log file types including 'csv', 'txt', 'tensorboard'
        :param tag: Additional tag of the log
        :param continue_log: Enable True when the log will be write to the base_dir
        '''
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        if continue_from_step is not None:
            self.log_base_dir = base_dir
        else:
            self.log_base_dir = os.path.join(base_dir, current_time + '_' + socket.gethostname() + '_' + tag)

        # set the continue log
        self.continue_from_step = continue_from_step if continue_from_step is not None and continue_from_step > 0 else 0

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if not os.path.exists(self.log_base_dir):
            os.mkdir(self.log_base_dir)

        # check if metadata exsists
        self.meta_file_path = os.path.join(self.log_base_dir, 'meta.json')
        if os.path.exists(self.meta_file_path):
            with open(self.meta_file_path) as json_data:
                self.meta_dict = json.load(json_data)
                json_data.close()
                self.meta_dict['description'] += '#[' + current_time + '_' + socket.gethostname() + ']:\n' + description + '\n'
                self.meta_dict['comment'] += '#[' + current_time + '_' + socket.gethostname() + ']:\n' + tag + '\n'
                if ckpt_path is not None:
                    self.meta_dict['history'] += '#[' + current_time + '_' + socket.gethostname() + ']:\n' + 'Continue From %s' % ckpt_path + '\n'
        else:
            # new meta data
            self.meta_dict = dict()
            self.meta_dict['history'] = '#[' + current_time + '_' + socket.gethostname() + ']:\n' + 'start initial training' + '\n'
            self.meta_dict['description'] = '#[' + current_time + '_' + socket.gethostname() + ']:\n' + description + '\n'
            self.meta_dict['comment'] = '#[' + current_time + '_' + socket.gethostname() + ']:\n' + tag  + '\n'
            self.meta_dict['lastest_step'] = 0

        self.log_types = log_types
        log_types_token = log_types.split('|')
        for log_type in log_types_token:
            log_type = log_type.strip()
            logger = self.logger_factory(log_type)
            if logger is not None:
                self.loggers[log_type] = logger

        # set the event handle
        atexit.register(self.close)
        signal.signal(signal.SIGTERM, self.__sig_handler__)

    def add_keys(self, keys):
        for log_type, logger in self.loggers.items():
            logger.add_keys(keys)

    def log(self, log_dict):
        log_dict['Iteration'] += self.continue_from_step
        for log_type, logger in self.loggers.items():
            logger.log(log_dict)

    def get_logger_by_type(self, type):
        if type in self.loggers:
            return self.loggers[type]
        else:
            return None

    def logger_factory(self, logger_name):
        if logger_name == "csv":
            return FileLogger(os.path.join(self.log_base_dir, 'log.csv'))
        elif logger_name == "txt":
            return FileLogger(os.path.join(self.log_base_dir, 'log.txt'))
        elif logger_name == "tensorboard" and self.continue_from_step > 0:
            return TensorboardLogger(self.log_base_dir, purge_step=self.continue_from_step)
        elif logger_name == "tensorboard":
            return TensorboardLogger(self.log_base_dir)
        else:
            return None

    def draw_architecture(self, model, input_shape, verbose=False):
        if 'tensorboard' in self.loggers.keys():

            writer = self.loggers['tensorboard'].writer

            if torch.cuda.is_available():
                dtype = torch.cuda.FloatTensor
            else:
                dtype = torch.FloatTensor

            # check if there are multiple inputs to the network
            if isinstance(input_shape[0], (list, tuple)):
                x = [Variable(torch.rand(1, *in_size)).type(dtype) for in_size in input_shape]
            else:
                x = Variable(torch.rand(1, *input_shape)).type(dtype)

            # draw the graph
            writer.add_graph(model, (x, ), verbose=verbose)

        else:
            warnings.warn('No instance of tensorboard logger configured')

    def flush(self):
        for log_type, logger in self.loggers.items():
            logger.flush()
    
    def close(self):
        self.save_meta_info()

    def print_meta_info(self):
        if self.meta_dict is not None:
            for meta_key in self.meta_dict.keys():
                print('--- ' + meta_key + ' -----\n' + str(self.meta_dict[meta_key]))

    def save_meta_info(self, add_log_dict=None):
        # get the current iteration
        cur_iteration = 0
        for log_type, logger in self.loggers.items():
            cur_iteration = logger.cur_iteration
            if cur_iteration > 0:
                break
            
        self.meta_dict['lastest_step'] = cur_iteration
        if add_log_dict is not None:
            current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
            for add_key in add_log_dict.keys():
                if add_key in self.meta_dict:
                    if add_key == 'history':
                        self.meta_dict[add_key] += '#[' + current_time + '_' + socket.gethostname() \
                                                   + ']:\n' + str(add_log_dict[add_key]) + '\n'
                    else:
                        self.meta_dict[add_key] += add_log_dict[add_key]
                else:
                    self.meta_dict[add_key] = add_log_dict[add_key]

        with open(self.meta_file_path, "w") as json_file:
            json.dump(self.meta_dict, json_file, indent=2)

    def __sig_handler__(self, signo, frame):
        print('[Terminated] See you next time!')
        sys.exit(0)
