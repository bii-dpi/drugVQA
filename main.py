import torch
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from trainAndTest import *

losses,accs,testResults = train(trainArgs)
    
