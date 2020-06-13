import os
import torch
import datetime
from collections import OrderedDict

class FileLogger:

    keys = ['Time', 'Event']

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file_type = log_file_path.split('.')[-1]
        self.header_flag = False
        self.line_count =0
        self.cur_iteration = -1

    def add_keys(self, keys):
        if isinstance(keys, list):
            for key in keys:
                if key not in self.keys:
                    self.keys.append(key)
        elif keys not in self.keys:
            self.keys.append(keys)

    def log(self, log_dict):

        current_time = datetime.datetime.now()
        self.cur_iteration = log_dict['Iteration']

        with open(self.log_file_path, 'a') as f:
            # Write to csv file
            if self.log_file_type == "csv":
                if self.header_flag is False:
                    for key in self.keys:
                        f.write(key + ',')
                    f.write('\n')
                    self.header_flag = True
                    self.line_count += 1

                # Write the line
                for key in self.keys:
                    if key == 'Time':
                        f.write(current_time.strftime("%Y-%m-%d %a %H:%M:%S") + ',')
                        continue

                    if key.startswith('Image'):
                        continue

                    if key.startswith('net'):
                        continue

                    if key in log_dict and not isinstance(log_dict[key], torch.nn.Module):
                        f.write(str(log_dict[key]) + ',')
                    else:
                        f.write(',')
                f.write('\n')
                self.line_count += 1

            # Write to txt file
            else:
                for key in self.keys:
                    if key == 'Time':
                        f.write(current_time.strftime("Time: " + "%Y-%m-%d %a %H:%M:%S") + '| ')
                        continue

                    if key in log_dict:
                        f.write(str(key) + ": " + str(log_dict[key]) + '| ')
                    else:
                        f.write("")
                f.write('\n')
                self.line_count += 1

    def flush(self):
        pass