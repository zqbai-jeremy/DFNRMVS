# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import shutil
from torch.autograd import Variable
from collections import OrderedDict
from collections import deque


def get_learning_rate(optimizer=None):
    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoints(file_path):
    return torch.load(file_path, map_location='cpu')


def assign_layer_tags(module):
    q = deque()

    # Add first submodules to queue
    for (name, module) in module._modules.items():
        setattr(module, 'tag', name)
        q.append(module)

    while len(q) != 0:
        front_module = q.popleft()
        module_tag = getattr(front_module, 'tag')
        for (name, submodule) in front_module._modules.items():
            setattr(submodule, 'tag', module_tag + '.' + name)
            q.append(submodule)


def create_module_tag_dict(module):
    q = deque()
    module_dict = {}

    # Add first submodules to queue
    for (name, module) in module._modules.items():
        if hasattr(module, 'tag'):
            module_dict[getattr(module, 'tag')] = module
        q.append(module)

    while len(q) != 0:
        front_module = q.popleft()
        module_tag = getattr(front_module, 'tag')
        for (name, submodule) in front_module._modules.items():
            if hasattr(submodule, 'tag'):
                module_dict[getattr(submodule, 'tag')] = module
            q.append(submodule)

    return module_dict


def summary_layers(model, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            if hasattr(module, 'tag'):
                module_tag = getattr(module, 'tag')
            else:
                module_tag = ''

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['type'] = class_name
            summary[m_key]['idx'] = module_idx
            summary[m_key]['tag'] = module_tag
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
                not isinstance(module, nn.ModuleList) and
                not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to add_graphthe network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size)).type(dtype)

    # print(type(x[0]))
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    # print(x.shape)
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    print('-------------------------------------------------------------------------------------------------')
    line_new = '{:>20} {:>20} {:>10} {:>25} {:>15}'.format('Type', 'Tag', 'Index', 'Output Shape', 'Param #')
    print(line_new)
    print('=================================================================================================')
    total_params = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = '{:>20} {:>20} {:>10} {:>25} {:>15}'.format(summary[layer]['type'],
                                                               summary[layer]['tag'],
                                                               str(summary[layer]['idx']),
                                                               str(summary[layer]['output_shape']),
                                                               summary[layer]['nb_params'])
        total_params += summary[layer]['nb_params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        print(line_new)
    print('=================================================================================================')
    print('Total params: ' + str(total_params))
    print('Trainable params: ' + str(trainable_params))
    print('Non-trainable params: ' + str(total_params - trainable_params))
    print('-------------------------------------------------------------------------------------------------')
    # return summary
