import os
import yaml
import torch

from utils import get_local_time


class Config:
    def __init__(self, config_dict=None):
        self._load_yamls()
        self._init_device()
        self.config_dict.update(config_dict)
        self._set_default_parameters()

    def _load_yamls(self):
        self.config_dict = {}
        current_path = os.path.dirname(__file__)
        overall_config_file = os.path.join(current_path, 'yamls/overall.yaml')
        model_config_file = os.path.join(current_path, 'yamls/model.yaml')
        dataset_config_file = os.path.join(current_path, 'yamls/dataset.yaml')

        for file in [overall_config_file, model_config_file, dataset_config_file]:
            with open(file, 'r', encoding='utf-8') as f:
                self.config_dict.update(yaml.load(f.read(), Loader=yaml.FullLoader))

    def _init_device(self):
        if self.config_dict['use_gpu']:
            gpu_id = self.config_dict['gpu_id']
            self.config_dict['device'] = torch.device(f'cuda:{gpu_id}'
                                                      if torch.cuda.is_available() else 'cpu')
        else:
            self.config_dict['device'] = torch.device('cpu')

    def _set_default_parameters(self):
        self.config_dict['filename'] = 'Fire-At-{}'.format(get_local_time())

    def __getitem__(self, item):
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            return None

    def __str__(self):
        args_info = '\n\nHyper Parameters:\n'
        for key, value in self.config_dict.items():
            args_info += '{}={}\n'.format(key, value)
        args_info += '\n'
        return args_info

    def __repr__(self):
        return self.__str__()
