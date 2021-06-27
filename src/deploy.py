import argparse

from subprocess import run

parser = argparse.ArgumentParser()

parser.add_argument("mode", type=str)
parser.add_argument("RV_SEED_INDEX", type=int)
parser.add_argument("SEED_INDEX", type=int)
parser.add_argument("FOLD_TYPE", type=str)
parser.add_argument("FOLD_NUM", type=int)
parser.add_argument("CUDA_NUM", type=int)

args = parser.parse_args()

run((f"python {args.mode}.py "
            f"{args.RV_SEED_INDEX} "
            f"{args.SEED_INDEX} "
            f"{args.FOLD_TYPE} "
            f"{args.FOLD_NUM} "
            f"{args.CUDA_NUM}").split())

