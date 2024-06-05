from .models import *
from .criterions import *
from .tasks import *
from .datasets import *

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

print("fairseq plugins loaded...")

import os
import importlib

# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("ctc_unity." + file_name)
