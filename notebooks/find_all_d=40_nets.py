import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from ast import literal_eval

def mathematica_import(s):
    s = s.replace('{','[')
    s = s.replace('}',']')
    l = literal_eval(s)
    return l

# DATA

# 1 HIDDEN LAYER
RAW_H1 = "{{2, 8}, {4, 4}, {6, 2}}"

#2 HIDDEN LAYERS
RAW_H2 = "{{2, 2, 7}, {2, 4, 4}, {2, 7, 2}, {4, 2, 4}}"

#3 HIDDEN LAYERS
RAW_H3 = "{{2, 2, 2, 6}, {2, 2, 6, 2}, {2, 3, 2, 5}, {2, 4, 2, 4}, {2, 5, 2, 3}, {2, 6, 2, 2}, {3, 3, 3, 2}, {3, 4, 2, 2}, {4, 2, 2, 3}, {4, 2, 3, 2}}"

RAW_H4 = "{{2, 2, 2, 2, 5}, {2, 2, 2, 5, 2}, {2, 2, 3, 2, 4}, {2, 2, 4, 2, 3}, {2, 2, 5, 2, 2}, {2, 3, 2, 2, 4}, {2, 3, 2, 4, 2}, {2, 4, 2, 2, 3}, {2, 4, 2, 3, 2}, {2, 5, 2, 2, 2}, {4, 2, 2, 2, 2}}"

RAW_H5 = "{{2, 2, 2, 2, 2, 4}, {2, 2, 2, 2, 4, 2}, {2, 2, 2, 3, 2, 3}, {2, 2, 2,4, 2, 2}, {2, 2, 3, 2, 2, 3}, {2, 2, 3, 2, 3, 2}, {2, 2, 4, 2, 2, 2}, {2, 3, 2, 2, 2, 3}, {2, 3, 2, 2, 3, 2}, {2, 3, 2, 3, 2, 2}, {2, 4, 2, 2, 2, 2}}"

RAW_H6 = "{{2, 2, 2, 2, 2, 2, 3}, {2, 2, 2, 2, 2, 3, 2}, {2, 2, 2, 2, 3, 2, 2}, {2, 2, 2, 3, 2, 2, 2}, {2, 2, 3, 2, 2, 2, 2}, {2, 3, 2, 2, 2, 2,2}}"

RAW_H7 = "{{2, 2, 2, 2, 2, 2, 2, 2}}"