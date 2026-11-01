"""Run from anywhere after extracting the add-on into the original project."""
import subprocess
import sys
from pathlib import Path

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    meshes=sorted((root/'outputs/fresh_test_cfd_v2').glob('N01/VTK/*/internal.vtu'))
    if len(meshes)!=1:
        raise FileNotFoundError('Expected N01 native mesh in the analysis archive')
    def run(*args): subprocess.run([sys.executable,*map(str,args)],cwd=root,check=True)
    run('src/train_deeponet_sst.py','--snapshots','outputs/native_sst/native_sst_snapshots.npz',
        '--mesh-path',meshes[0],'--output-dir','outputs/deeponet_sst',
        '--hidden-units','32','32','--activation','tanh','--optimizer','lbfgs',
        '--epochs','20','--l2','0.0001','--seed','42')
    run('src/cross_validate_deeponet.py','--snapshots','outputs/native_sst/native_sst_snapshots.npz',
        '--output-dir','outputs/deeponet_cv')
    run('src/evaluate_deeponet.py')
    run('src/report_deeponet.py')
    run('-m','unittest','discover','-s','tests','-v')
