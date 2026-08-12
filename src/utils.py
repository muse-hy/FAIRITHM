# Source: https://github.com/CharlesYu2000/PCGU-UnlearningBias (src/utils.py)
# License: CC BY-SA 4.0 — redistributed with attribution; see the original
# repository for the license text. Required by src/general_similarity_retrain_ko.py.
import numpy as np
import torch
import random

def set_random_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
