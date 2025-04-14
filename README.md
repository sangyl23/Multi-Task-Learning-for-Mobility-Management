# multi task learning for mobility management
This is the code for our IEEE WCL ``Dual-Cascaded Multi-Task Learning for Mobility Management in mmWave Wireless Systems''

## Reproduce the experimental result

In our experiments, in addition to the simulations described in the original letter, we also provide simulations for different millimeter-wave scenarios, different motion forms, different signal-to-noise ratios of the beam received signals, and different user velocities.

* For Neural ODE, please run

```
python spiral.py --adjoint=1 --visualize=1 --niters=10000 --model_name Neural_ODE --noise_a=0.02 --cc=2 --train_dir ./sprial_neuralode
```

* For ContiFormer, please run

```
python spiral.py --adjoint=1 --visualize=1 --niters=10000 --model_name Contiformer --noise_a=0.02 --cc=2 --train_dir ./spiral_contiformer
```

The results and visualization data will be saved to `./sprial_neuralode` and `./spiral_contiformer`. 

All the experimental results are averaged, i.e., `27, 42, 1024` (the same random seeds for other tasks), use `--seed` to set the seed.
