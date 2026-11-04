# Earlier RANS POD–MLP

Core training and inference code: `src/pod_naca0012.py`, including
`train_physics_informed` and `predict_physics_case`. Saved POD, turbulent
viscosity basis, trained checkpoint, validation predictions, loss history, and
summary are in `outputs/physics_pod_model/`. The branch inherits the original
common-grid training and validation arrays from `main`.

This model adds approximate steady mean-flow RANS losses to the 4-mode
`u,w,p` surrogate. It does not enforce the k–omega SST transport equations.
Its seven-case validation errors are u 0.862%, w 1.592%, p 3.737%.
Raw CFD exports are still required to retrain the turbulent-viscosity support
model and physics terms from scratch.
