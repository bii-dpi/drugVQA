import torch
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from trainAndTest import *

#print(torch.cuda.device_count())
#torch.cuda.set_device(2)
#print(torch.cuda.current_device())

train(trainArgs)
    
