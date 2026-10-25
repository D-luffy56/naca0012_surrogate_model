"""Predict coefficients and dimensional fields without CFD coefficient inputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from train_deeponet_sst import predict_sst_case

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,default=Path('outputs/deeponet_sst'))
    parser.add_argument('--re',type=float,required=True)
    parser.add_argument('--aoa',type=float,required=True)
    parser.add_argument('--data-only',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('outputs/deeponet_prediction'))
    args=parser.parse_args()
    pred=predict_sst_case(args.model_dir,args.re,args.aoa,physics=not args.data_only)
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'mode':np.arange(1,len(pred['coefficients'])+1),
                  'coefficient':pred['coefficients']}).to_csv(args.output/'pod_coefficients.csv',index=False)
    np.savez_compressed(args.output/'fields.npz',**pred,reynolds=args.re,aoa_deg=args.aoa)
    print(args.output/'pod_coefficients.csv')
