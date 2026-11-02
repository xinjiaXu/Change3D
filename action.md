# Change3D Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/Change3D`  
Dataset root: `/home/xxj/Data`

## Pull reproduce branch

First time on the server:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/Change3D
git fetch origin
git switch --track -c reproduce origin/reproduce
```

Later updates:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/Change3D
git switch reproduce
git pull --ff-only
```

## Environment and pretrained weight

```bash
bash setup_server_env.sh
conda activate change3d-rs
mkdir -p model
wget -O model/X3D_L.pyth https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth
bash prepare_server_data.sh /home/xxj/Data
```

## Model complexity

After activating `change3d-rs`, install the profiling dependency and count parameters/FLOPs for one paired 256x256 input:

```bash
python -m pip install -r requirements_profile.txt
CUDA_VISIBLE_DEVICES=0 python profile_model.py --device cuda:0 \
  --output Change3D_work_dirs/profile_256.json
```

The JSON reports total/trainable parameters and fvcore-counted operations. Unsupported operators are listed; the FLOP total is incomplete if that list is non-empty. Profiling initializes the BCD architecture without loading the X3D checkpoint.

## Train and test

`run_bcd.sh` trains and then tests, using the existing defaults. Run one dataset at a time:

The latest `reproduce` branch also allows CDD, SYSU-CD, and LEVIR-CD-PLUS in the BCD logger. If the server still reports `setup_logger error`, update the branch before starting:

```bash
git switch reproduce
git pull --ff-only
```

```bash
bash run_bcd.sh levircd
bash run_bcd.sh levircdplus
bash run_bcd.sh cdd
bash run_bcd.sh whucd
bash run_bcd.sh sysucd
```

Select GPU 1 or adjust workers with `GPU_ID=1 NUM_WORKERS=8 bash run_bcd.sh levircd`. Outputs are in `Change3D_work_dirs/<dataset>/`.

**Evaluation note:** the BCD loop uses the test loader to choose `best_model.pth` each epoch, then evaluates that same split at the end. The resulting test score is not an independent held-out score.
