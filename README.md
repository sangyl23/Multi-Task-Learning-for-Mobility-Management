# multi task learning for mobility management
This is the code for our IEEE WCL ``Dual-Cascaded Multi-Task Learning for Mobility Management in mmWave Wireless Systems''.

## File Description
 - `./DeepMIMO_O1_28_matlab` containing matlab code for generating Outdoor 1 dataset.
 - `./DeepMIMO_O1_28B_matlab` containing matlab code for generating Outdoor 1 Blockage dataset.
 - `./eval_dataset` test data set folder.
 - `./trained_model` trained model folder.
 - `./our_results` This folder contains the results generated by our previously executed programs.
 - `dataloader_MTL.py` code for data loading.
 - `model_MTL.py` code containing all function templates for models. Specifically, `Bs`, `bt`, and `Up` represent the single-task learning models for BS prediction, beam tracking, and UE positioning, respectively. `Vanilla` denotes the vanilla multi-task learning model. `Bs2bt2Up` is the single-cascaded multi-task learning model. `Up2bt2Bs` is the inversed single-cascaded multi-task learning model. `Dual_Cascaded` is the proposed dual-cascaded multi-task learning model.
 - `eval_MTL.py` eval different models.

## Keyword arguments
In our code, we have provided detailed comments. Below are the specific meanings of some keywords:
 - `b` batch size.
 - `his_len` historical sequence length.
 - `BS_num` BS number.
 - `beam_num` beam number.

## Date set generation
In our code, we have provided the data set for test. If you want to reproduce the data set, please follow these steps:

### Outdoor 1 
 1. Download the original ray-tracking dataset from https://www.deepmimo.net/scenarios/o1-scenario/.
 2. Run `Generator_Originaldata.mat`.
 3. If you want UE performs the rectilinear motion, run `Generator_MTL.mat`. If you want UE performs the spiral motion, run `Generator_MTL_spiral_2D.mat`.
 You can adjust UE velocities and signal-to-noise ratios (SNRs) of the beam received signals in `Generator_MTL.mat`.

### Outdoor 1 
 1. Download the original ray-tracking dataset from https://www.deepmimo.net/scenarios/o1-scenario/.
 2. Run `Generator_Originaldata.mat`.
 3. If you want UE performs the rectilinear motion, run `Generator_MTL.mat`. If you want UE performs the spiral motion, run `Generator_MTL_spiral_2D.mat`.

## Reproduce the experimental result

In our experiments, in addition to the simulations described in the original letter, we also provide the simulations for different UE velocities, different SNRs of the beam received signals, different millimeter-wave scenarios, and different motion forms.

* For the simulations described in the original letter, please run

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

Note: Due to GitHub's storage limitations, we can only upload a single trained model. The simulation results in the original letter were averaged over multiple models, which may lead to minor numerical differences between the original letter and this pytorch implement. However, this does not affect the conclusions drawn in the original letter.

## Reference

You are more than welcome to cite our paper:
```
@inproceedings{sang2025dualcascadedmtl,
  title={Dual-Cascaded Multi-Task Learning for Mobility Management in mmWave Wireless Systems},
  author={Yiliang Sang, Yingshuang Bai, Ke Ma, Chen Sun, Pengyu Wang, Lebin Yao, and Zhaocheng Wang},
  booktitle={IEEE WCL},
  year={2025}
}
```

