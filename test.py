import torch
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from trainAndTest import *

for i in range(1, 25):
    curr_path = f"model_pkl/DUDE/Complete-{i}.pkl"
    testArgs['model'] = DrugVQA(modelArgs,block = ResidualBlock).cuda()
    testArgs['model'].load_state_dict(torch.load(curr_path))
    print(test(testArgs))
