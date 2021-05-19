import torch
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from trainAndTest import *
from progressbar import progressbar


testArgs['saveNamePre'] = "cv_3_"

print("Validating...")
for i in progressbar(range(40, 30, -2)):
    curr_path = f"../model_pkl/DUDE/{testArgs['saveNamePre']}{i}.pkl"
    testArgs['model'] = DrugVQA(modelArgs,block = ResidualBlock).cuda()
    testArgs['model'].load_state_dict(torch.load(curr_path))
    test(testArgs, i)

print("Finished validation.")
