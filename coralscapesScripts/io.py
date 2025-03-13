import yaml
import argparse

class ConfigDict(dict):
    """
    A dictionary subclass that allows attribute-style access to dictionary keys.
    This class recursively converts nested dictionaries into ConfigDict instances,
    enabling dot notation access to dictionary keys.
    Methods
    -------
    __init__(dictionary)
        Initializes the ConfigDict with the given dictionary, converting nested
        dictionaries to ConfigDict instances.
    __getattr__(attr)
        Allows access to dictionary keys as attributes.
    __setattr__(key, value)
        Allows setting dictionary keys as attributes.
    """

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigDict(value)
            self[key] = value

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)
    

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config(base_config, config):
    updated_config = base_config.copy()
    for key, value in config.items():
        if key in updated_config:
            if isinstance(value, dict):
                updated_config[key].update(value)
            else:
                updated_config[key] = value
        else:
            updated_config[key] = value
    return updated_config

def setup_config(config_path=None, config_base_path = "config/base.yaml"):
    base_config = load_config(config_base_path)
    if(config_path is None):
        return ConfigDict(base_config)
    config = load_config(config_path)
    updated_config = update_config(base_config, config)
    return ConfigDict(updated_config)

def get_parser():
    parser = argparse.ArgumentParser(description='Semantic Segmentation on the Coralscapes Dataset')

    # model and dataset
    parser.add_argument("--run-name", type=str)
    parser.add_argument('--wandb-project', type=str, 
                        help='wandb project name')
    parser.add_argument('--model', type=str,
                        choices=['deeplabv3+resnet101', 'deeplabv3+resnext50', 'fcn_resnet101', 'deeplabv3+resnext50_32x4d'],
                        help='model name')
    parser.add_argument('--weight', type=bool,
                        help='add weights to loss')
    
    # training hyper params
    parser.add_argument('--batch-size', type=int,
                        help='input batch size for training')
    parser.add_argument('--batch-size-eval', type=int,
                        help='input batch size for evaluation')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float,
                        help='learning rate')
    
    # lora hyper params
    parser.add_argument('--r', type=int,
                        help='LoRA rank')
    
    # checkpoint and log
    parser.add_argument('--model-checkpoint', type=str,
                        help='path to model checkpoint')
    
    parser.add_argument('--log-epochs', type=int,
                        help='log every log-epochs')

    parser.add_argument('--config', type=str, 
                        help='path to config file')
    
    parser.add_argument("--inputs", type=str, help="path to directory with images")
    parser.add_argument("--outputs", type=str, help="path to directory to save outputs")

    return parser

def update_config_with_args(config, args):
    if args.run_name:
        config['run_name'] = args.run_name
    if args.model:
        config['model']['name'] = args.model
    if(args.model_checkpoint):
        config['model']['checkpoint'] = args.model_checkpoint
    if args.weight:
        config['data']['weight'] = args.weight
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.batch_size_eval:
        config['data']['batch_size_eval'] = args.batch_size_eval
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['optimizer']['lr'] = args.lr
    if args.r:
        config['lora']['r'] = args.r
    if args.wandb_project:
        config['logger']['wandb_project'] = args.wandb_project
    if args.log_epochs:
        config['logger']['log_epochs'] = args.log_epochs

    return config


