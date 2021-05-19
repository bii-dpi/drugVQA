import torch
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from trainAndTest import *

if __name__ == "__main__":
    mp.spawn(train, nprocs=args.gpus, args=(trainArgs,))
