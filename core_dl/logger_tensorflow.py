import datetime
import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


class TensorboardLogger:

    keys = ['Time', 'Event', 'Iteration', 'net']

    def __init__(self, log_file_dir=None, purge_step=None):
        self.log_file_dir = log_file_dir
        self.writer = SummaryWriter(logdir=log_file_dir, purge_step=purge_step)
        self.enable_param_histogram = True
        self.cur_iteration = -1

    def __del__(self):
        self.writer.close()

    def add_keys(self, keys):
        if isinstance(keys, list):
            for key in keys:
                if key not in self.keys:
                    self.keys.append(key)
        elif keys not in self.keys:
            self.keys.append(keys)

    def attach_layer_params(self, net):
        self.enable_param_histogram = True
        for name, param in net.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.cur_iteration)

    def log(self, log_dict):
        self.cur_iteration = log_dict['Iteration']
        for key in self.keys:
            if key in log_dict:

                # Write the Scalar to tensorboard
                if key.startswith('Loss') or key.startswith('Accuracy') or key.startswith('Scalar'):
                    self.writer.add_scalar(key, log_dict[key], self.cur_iteration)

                # Write the Scalars to tensorboard
                if key.startswith('Scalars'):
                    # The log_dict[key] should be another dict, e.g.
                    # log_dict[key]= {'xsinx': n_iter * np.sin(n_iter),
                    #                 'xcosx': n_iter * np.cos(n_iter),
                    #                 'arctanx': np.arctan(n_iter)}
                    self.writer.add_scalars(key, log_dict[key], self.cur_iteration)

                if key.startswith('Histogram'):
                    self.writer.add_histogram(key, log_dict[key].cpu().detach().numpy(), self.cur_iteration)

                # Update related statistic info of network structure (and parameters)
                if isinstance(log_dict[key], torch.nn.Module) and self.enable_param_histogram:
                    # Update the parameter histogram
                    for name, param in log_dict[key].named_parameters():
                        self.writer.add_histogram('param_' + name, param.clone().cpu().data.numpy(), self.cur_iteration)
                        self.writer.add_histogram('grad_' + name, param.grad.clone().cpu().data.numpy(), self.cur_iteration)

                # Write the Image to tensorboard
                if key.startswith('Image'):
                    # The log_dict[key] should be a list of image tensors
                    img_grid = log_dict[key]
                    if len(log_dict[key]) > 1:
                        img_grid = make_grid(img_grid, nrow=1)
                    else:
                        img_grid = img_grid[0]
                    self.writer.add_image(key, img_grid, self.cur_iteration)

    def flush(self):
        pass
