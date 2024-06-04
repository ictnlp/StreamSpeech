from .models import *
from .criterions import *
from .tasks import *
from .datasets import *

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

print("fairseq plugins loaded...")
