"""Reproducible comparison on train, existing validation and additional CFD cases.

No model selection uses validation/test errors. All reported norms use pointwise
Euclidean weighting, matching the original code (not cell-volume weighting).
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import meshio
import torch
from torch import nn
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coefficient_deeponet import CoefficientDeepONet, dense
from train_deeponet_sst import graph_derivatives, as_torch_sparse
from sst_deeponet_physics import make_residuals

FIELDS = ('u', 'w', 'p', 'k', 'omega', 'nut')


def read_npz(path):
    with np.load(path) as d:
        return {k:d[k].copy() for k in d.files}


def extract(mesh):
    points, index = np.unique(mesh.points[:, [0,2]], axis=0, return_index=True)
    d = mesh.point_data
    return points, np.stack((d['U'][:,0],d['U'][:,2],d['p'],d['k'],d['omega'],d['nut']))[:,index]


def decode(c, b):
    return (b['mean'] + c @ b['modes']).reshape(-1,6,len(b['points_xz'])) * b['scale'][None,:,None]


def project(q, b):
    return ((q / b['scale'][None,:,None]).reshape(len(q),-1)-b['mean']) @ b['modes'].T


def net_predict(net, x, b):
    # Preserve float32 StandardScaler semantics used in the original trainer.
    z = np.asarray(x,dtype=np.float32).copy()
    z -= b['x_mean']; z /= b['x_scale']
    with torch.no_grad():
        return net(torch.tensor(z)).numpy()*b['coefficient_scale']+b['coefficient_mean']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model-dir', type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    root=args.root; out=args.model_dir or root/'outputs/deeponet_sst'
    result=out/'evaluation'; result.mkdir(exist_ok=True)
    d=read_npz(root/'outputs/native_sst/native_sst_snapshots.npz')
    b=read_npz(out/'sst_pod_basis.npz'); count=int(d['train_count'])
    v=d['uinf_mps']; units=np.stack((v,v,v*v,v*v,v,v),axis=1)
    q=d['fields']/units[:,:,None]
    wall=d['wall_fields']/units[:,:,None]
    x=np.column_stack((d['reynolds'],d['aoa_deg']))
    all_coef=project(q,b)
    models={}
    for name, filename in [('DeepONet-data','deeponet_data.pt'),('DeepONet-SST','deeponet_sst.pt')]:
        net=CoefficientDeepONet(len(b['modes']),tuple(b['hidden_units']),str(b['activation']),int(b['latent']))
        net.load_state_dict(torch.load(out/filename,map_location='cpu',weights_only=True)); net.eval()
        models[name]=(net,b)
    # New paired MLP uses exactly this run's basis, targets, scaling and seed.
    xz=((x[:count]-b['x_mean'])/b['x_scale']).astype(np.float32)
    yz=((all_coef[:count]-b['coefficient_mean'])/b['coefficient_scale']).astype(np.float32)
    mlp=MLPRegressor(hidden_layer_sizes=(32,32),activation='tanh',solver='lbfgs',
                     alpha=1e-4,max_iter=5000,random_state=42).fit(xz,yz)
    class LegacyNet(nn.Module):
        def __init__(self, rank, hidden, activation):
            super().__init__(); self.layers=dense((2,*hidden,rank),activation)
        def forward(self,x): return self.layers(x)
    for name, folder, hidden, act in [('Saved-MLP64-SST','sst_pod_model',(64,64),'relu'),
                                    ('Saved-MLP32-SST','sst_pod_tanh32_lbfgs_l2_1e-4',(32,32),'tanh')]:
        directory=root/'outputs'/folder; old=read_npz(directory/'sst_pod_basis.npz')
        net=LegacyNet(len(old['modes']),hidden,act)
        net.load_state_dict(torch.load(directory/'sst_pod_mlp.pt',map_location='cpu',weights_only=True)); net.eval()
        models[name]=(net,old)
    testdir=root/'outputs/fresh_test_cfd_v2'
    specs=pd.read_csv(testdir/'frozen_cases.csv'); test=[]; meshpath=None
    for row in specs.itertuples():
        paths=list((testdir/row.id/'VTK').rglob('internal.vtu'))
        if len(paths)!=1: raise ValueError(f'Expected one export for {row.id}')
        meshpath=paths[0] if meshpath is None else meshpath
        pts, raw=extract(meshio.read(paths[0]))
        if pts.shape!=d['points_xz'].shape or not np.allclose(pts,d['points_xz'],atol=1e-7,rtol=0):
            raise ValueError('Test mesh mismatch')
        speed=row.Re*1.56e-5
        test.append(raw/np.array((speed,speed,speed**2,speed**2,speed,speed))[:,None])
    datasets={'training':(q[:count],x[:count],d['ids'][:count],wall[:count]),
              'validation':(q[count:],x[count:],d['ids'][count:],wall[count:]),
              'additional_test':(np.stack(test),specs[['Re','AoA_deg']].to_numpy(),specs.id.to_numpy(),None)}
    Dx,Dz,valid=graph_derivatives(meshpath,d['points_xz'],d['wall_distance'])
    residuals=make_residuals(as_torch_sparse(Dx),as_torch_sparse(Dz),d['wall_distance'])
    with torch.no_grad():
        ref=residuals(torch.tensor(q[:count],dtype=torch.float32),torch.tensor(x[:count,0],dtype=torch.float32))
        scales={k:float(v[:,valid].square().mean().sqrt().clamp_min(1e-5)) for k,v in ref.items()}
    points=d['points_xz']; distance=d['wall_distance']; wi=d['wall_index']
    regions={'all':np.ones(len(points),bool),'near_wall':distance<=0.05,
             'wake':(points[:,0]>=1)&(points[:,0]<=3)&(abs(points[:,1])<=0.5)}
    summary={'protocol':'40 training only; fixed settings before evaluation. Validation and additional cases were previously inspected in the project, so neither is a newly untouched test set.',
             'physics_reference_scales':scales,'datasets':{},
             'coefficient_note':'All coefficient metrics use this DeepONet run\'s 16-mode basis. Saved MLP fields are projected into it for coefficient comparison.',
             'additional_cfd_quality':json.loads((testdir/'comparison_summary.json').read_text())['quality']}
    field_rows=[]; coefficient_rows=[]; residual_rows=[]; parity={}
    for split,(truth,features,ids,wall_truth) in datasets.items():
        actual=project(truth,b)
        predicted={}; predicted_coef={}
        for name,(net,basis) in models.items():
            coeff=net_predict(net,features,basis); pred=decode(coeff,basis)
            predicted[name]=pred
            predicted_coef[name]=coeff if name.startswith('DeepONet') else project(pred,b)
        coeff=mlp.predict(((features-b['x_mean'])/b['x_scale']).astype(np.float32))*b['coefficient_scale']+b['coefficient_mean']
        predicted['MLP32-refit-data']=decode(coeff,b); predicted_coef['MLP32-refit-data']=coeff
        predicted['POD-projection']=decode(actual,b); predicted_coef['POD-projection']=actual
        predicted['CFD']=truth; predicted_coef['CFD']=actual
        group={}
        for name,pred in predicted.items():
            if not np.isfinite(pred).all(): raise FloatingPointError(name)
            pc=predicted_coef[name]; error=pc-actual
            record={'coefficient_rmse':float(np.sqrt(np.mean(error**2))),
                    'coefficient_mae':float(np.mean(abs(error))),
                    'coefficient_standardized_rmse':float(np.sqrt(np.mean((error/b['coefficient_scale'])**2))),
                    'coefficient_relative_l2':float(np.linalg.norm(error)/max(np.linalg.norm(actual),1e-12)),
                    'coefficient_r2_uniform':float(r2_score(actual,pc)), 'regions':{},
                    'negative_fraction':{f:float((pred[:,i]<0).mean()) for i,f in enumerate(FIELDS) if i>=3}}
            for region,mask in regions.items():
                record['regions'][region]={}
                for i,f in enumerate(FIELDS):
                    diff=pred[:,i,mask]-truth[:,i,mask]
                    record['regions'][region][f]={'relative_l2':float(np.linalg.norm(diff)/max(np.linalg.norm(truth[:,i,mask]),1e-12)),
                        'rmse_nondimensional':float(np.sqrt(np.mean(diff**2))),
                        'mae_nondimensional':float(np.mean(abs(diff)))}
                for j,case in enumerate(ids):
                    field_rows.append({'split':split,'model':name,'case':case,'region':region,
                        **{f:float(np.linalg.norm(pred[j,i,mask]-truth[j,i,mask])/max(np.linalg.norm(truth[j,i,mask]),1e-12)) for i,f in enumerate(FIELDS)}})
            if wall_truth is not None:
                record['wall_normalized_rmse']={f:float(np.sqrt(np.mean((pred[:,i,wi]-wall_truth[:,i])**2))/b['scale'][i]) for i,f in enumerate(FIELDS)}
                record['cp_rmse']=float(np.sqrt(np.mean((2*pred[:,2,wi]-2*wall_truth[:,2])**2)))
            rms={}
            with torch.no_grad():
                sums={k:0. for k in scales}
                for start in range(0,len(truth),4):
                    terms=residuals(torch.tensor(pred[start:start+4],dtype=torch.float32),torch.tensor(features[start:start+4,0],dtype=torch.float32))
                    for k,val in terms.items(): sums[k]+=float(val[:,valid].square().sum())
                rms={k:float(np.sqrt(v/(len(truth)*valid.sum()))) for k,v in sums.items()}
            record['residual_rms']=rms
            record['normalized_physics_score']=float(np.mean([(rms[k]/scales[k])**2 for k in rms]))
            residual_rows.append({'split':split,'model':name,**rms,'normalized_score':record['normalized_physics_score']})
            for j,case in enumerate(ids):
                for k in range(actual.shape[1]):
                    coefficient_rows.append({'split':split,'model':name,'case':case,'mode':k+1,'actual':actual[j,k], 'predicted':pc[j,k], 'error':error[j,k]})
            group[name]=record
        summary['datasets'][split]=group
        if split=='validation':
            # Same dimensional Cartesian targets as the original three-field MLP.
            legacy=root/'outputs/baseline_recomputed'
            oldbasis=read_npz(legacy/'pod_basis.npz')
            oldtruth=read_npz(legacy/'validation_snapshots.npz')
            oldpred=read_npz(legacy/'validation_predictions.npz')
            if not np.array_equal(oldtruth['ids'],ids) or not np.array_equal(oldpred['ids'],ids):
                raise ValueError('Legacy case IDs mismatch')
            xx,zz=np.meshgrid(oldbasis['x'],oldbasis['z'],indexing='xy')
            mask=oldbasis['fluid_mask'];target=np.column_stack((xx[mask],zz[mask]))
            tri=Delaunay(points); element=tri.find_simplex(target)
            if np.any(element<0): raise ValueError('Legacy grid outside native mesh')
            transform=tri.transform[element]
            bary=np.einsum('nij,nj->ni',transform[:,:2],target-transform[:,2])
            weights=np.column_stack((bary,1-bary.sum(axis=1)));vertices=tri.simplices[element]
            common={'Original-POD-MLP':oldpred['fields']}
            for name in ['DeepONet-data','DeepONet-SST','POD-projection']:
                dimensional=predicted[name][:,:3]*units[count:,:3,None]
                common[name]=(dimensional[:,:,vertices]*weights[None,None,:,:]).sum(axis=-1)
            summary['original_grid_validation']={name:{f:float(np.linalg.norm(value[:,i]-oldtruth['fields'][:,i])/np.linalg.norm(oldtruth['fields'][:,i])) for i,f in enumerate(FIELDS[:3])} for name,value in common.items()}
            parity=(actual,predicted_coef['DeepONet-data'],predicted_coef['DeepONet-SST'])
            np.savez_compressed(result/'coefficient_predictions.npz',ids=ids,actual=actual,
                                data=parity[1],sst=parity[2])
            for j,case in enumerate(ids):
                fig,ax=plt.subplots(figsize=(7,4))
                for side in [points[wi,1]>=0,points[wi,1]<0]:
                    order=np.argsort(points[wi[side],0]); idx=wi[side][order]
                    ax.plot(points[idx,0],2*wall_truth[j,2,side][order],color='black',lw=1)
                    for name,color in [('DeepONet-SST','tab:blue'),('MLP32-refit-data','tab:orange')]:
                        ax.plot(points[idx,0],2*predicted[name][j,2,idx],color=color,ls='--',label=name if side[0] else None)
                ax.set(xlabel='x/c',ylabel='Cp',title=f'{case}: CFD black; DeepONet-SST blue; MLP orange')
                ax.invert_yaxis();fig.tight_layout();fig.savefig(result/f'cp_{case}.png',dpi=140);plt.close(fig)
        print('Evaluated',split,flush=True)
    # Warm resident CPU inference: include coefficient scaling + six-field decoder.
    times={}
    for name in ['DeepONet-data','DeepONet-SST']:
        net,basis=models[name]
        for _ in range(5): decode(net_predict(net,x[:1],basis),basis)
        samples=[]
        for _ in range(50):
            start=time.perf_counter();decode(net_predict(net,x[:1],basis),basis);samples.append(time.perf_counter()-start)
        times[name]={'median_seconds':float(np.median(samples)),'p95_seconds':float(np.quantile(samples,.95))}
    summary['resident_cpu_inference']=times
    pd.DataFrame(field_rows).to_csv(result/'field_errors_by_case.csv',index=False)
    pd.DataFrame(coefficient_rows).to_csv(result/'coefficient_errors.csv',index=False)
    pd.DataFrame(residual_rows).to_csv(result/'physics_residuals.csv',index=False)
    (result/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,pred,label in zip(axes,parity[1:],['Data-only','Physics-informed']):
        actual=parity[0];ax.scatter(actual.ravel(),pred.ravel(),s=12,alpha=.6)
        lim=[min(actual.min(),pred.min()),max(actual.max(),pred.max())];ax.plot(lim,lim,'k--')
        ax.set(xlabel='Actual POD coefficient',ylabel='Predicted POD coefficient',title=label)
    fig.tight_layout();fig.savefig(result/'coefficient_parity.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,4.5))
    labels=['MLP32-refit-data','Saved-MLP32-SST','Saved-MLP64-SST','DeepONet-data','DeepONet-SST','POD-projection']
    for ax,split in zip(axes,['validation','additional_test']):
        for j,name in enumerate(labels):
            values=[100*summary['datasets'][split][name]['regions']['all'][f]['relative_l2'] for f in FIELDS]
            ax.bar(np.arange(6)+(j-2.5)*.13,values,.13,label=name)
        ax.set_xticks(np.arange(6),FIELDS);ax.set(ylabel='Relative L2 error (%)',title=split)
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(result/'field_comparison.png',dpi=150);plt.close(fig)

if __name__=='__main__': main()
