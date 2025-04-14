# multi task learning for mobility management
This is the code for our IEEE WCL ``Dual-Cascaded Multi-Task Learning for Mobility Management in mmWave Wireless Systems''.

## File Description

#### Keyword arguments:
 - `eval_dataset` test data set folder.
 - `trained_model` trained model folder.
 - `dataloader_MTL.py` code for data loading.
 - `model_MTL.py` code containing all function templates for models. Specifically, `Bs`, `bt`, and `Up` represent the single-task learning models for BS prediction, beam tracking, and UE positioning, respectively. `Vanilla` denotes vanilla multi-task learning model. `Vanilla` denotes vanilla multi-task learning model


## Reproduce the experimental result

In our experiments, in addition to the simulations described in the original letter, we also provide the simulations for different UE velocities, different signal-to-noise ratios (SNRs) of the beam received signals, different millimeter-wave scenarios, and different motion forms.

* For the described in the original letter, please run

```
python eval_MTL.py --experiment_type O1_trainingsamples
```

* For different UE velocities, please run

```
python eval_MTL.py --experiment_type O1_velocity
```

* For different SNRs of the beam received signals, please run

```
python eval_MTL.py --experiment_type O1_snr
```

* For different motion forms, we consider UE performs the 2D spiral motion, and you can run

```
python eval_MTL.py --experiment_type O1_motion_form
```

* For different millimeter-wave scenarios, we consider a more complexed millimeter-wave scenarios named ``Outdoor1 Blockage'' [1], and you can run

```
python eval_MTL.py --experiment_type Outdoor_Blockage_trainingsamples
```

The visualization figures will be saved to `./results`. 

