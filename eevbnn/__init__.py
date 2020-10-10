import sys
import torch
if sys.version_info < (3, 8):
    raise SystemError('at least python3.8 is required')

torch.manual_seed(0)    # for reproducibility
