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
