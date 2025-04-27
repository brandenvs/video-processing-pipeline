from .cli_args import args
from .folder_paths import *
from .model_management import get_torch_device
from .options import enable_args_parsing

SERVICES = {
    'GetDevice': get_torch_device
}