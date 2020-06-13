import torch.nn as nn
import torch
from core_dl.module_util import summary_layers


class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.net_name = self.__class__.__name__
        self.input_shape_chw = None

    def forward(self, *input):
        pass

    def load_state(self, state_dict):
        """
        Load the network from state_dict with the key name, can be used for loading part of pre-trained network
        :param state_dict: state_dict instance, use torch.load() to get the state_dict
        """
        cur_model_state = self.state_dict()
        input_state = {k: v for k, v in state_dict.items() if
                       k in cur_model_state and v.size() == cur_model_state[k].size()}
        cur_model_state.update(input_state)
        self.load_state_dict(cur_model_state)

    def save_net_def(self, dir):
        pass

    def summary(self):
        """
        Print network structure
        :return: none
        """
        print('Name:' + self.net_name)
        if torch.cuda.is_available():
            print("GPU: Enabled")

        if self.input_shape_chw is not None:
            print('Input Shape: ' + str(self.input_shape_chw))
            print('net Structure:')
            summary_layers(self, self.input_shape_chw)

    def find_module(self, key):
        if key in self.module_dict:
            return self.module_dict[key]
        else:
            return None
