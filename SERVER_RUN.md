# Change3D server setup

The helper scripts keep the upstream model, optimizer, loss, augmentation, learning-rate schedule, input size, and iteration budget unchanged. They create a separate environment, expose existing dataset folders through symlinks, and direct outputs to `Change3D_work_dirs/<dataset>/`.

## Server paths

- Repository: `/home/xxj/code/DINOCLIP_comparative_test/Change3D`
- Datasets: `/home/xxj/Data`
- Outputs: `Change3D_work_dirs/{levircd,levircdplus,cdd,whucd,sysucd}`

## Environment

From the repository directory:

```bash
bash setup_server_env.sh
conda activate change3d-rs
```

The README specifies Python 3.11 and PyTorch 2.2 with CUDA 11.8. That PyTorch/CUDA build predates RTX 5090 Blackwell support. The setup script uses Python 3.11 and PyTorch 2.7.1 CUDA 12.8 wheels in the isolated `change3d-rs` environment, then installs the upstream `requirements.txt`. It does not change the base environment.

## Pretrained X3D-L checkpoint

The upstream trainer expects a checkpoint containing the `model_state` key and defaults to `model/X3D_L.pyth`:

```bash
mkdir -p model
wget -O model/X3D_L.pyth https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth
```

## Dataset layout adaptation

Run:

```bash
bash prepare_server_data.sh /home/xxj/Data
```

This creates `data_adapters/<dataset>/{train,val,test}/{t1,t2,label}` as symlinks. Source samples are not copied, renamed, resized, or modified.

| Dataset key | Server source | Image folders | Label folder |
| --- | --- | --- | --- |
| `levircd` | `/home/xxj/Data/LEVIR_CD` | `A`, `B` | `label` |
| `levircdplus` | `/home/xxj/Data/LEVIR_CD_PLUS/cut_data` | `A`, `B` | `label` |
| `cdd` | `/home/xxj/Data/CDD` | `A`, `B` | `OUT` |
| `whucd` | `/home/xxj/Data/WHUCD` | `A`, `B` | `label` |
| `sysucd` | `/home/xxj/Data/SYSU_CD` | `time1`, `time2` | `label` |

Change3D's BCD loader expects lowercase `train`, `val`, and `test` directories containing `t1`, `t2`, and `label`. The links adapt the supplied directory names to that interface.

## Training

After activating the environment and preparing the links, run one dataset at a time:

```bash
bash run_bcd.sh levircd
bash run_bcd.sh levircdplus
bash run_bcd.sh cdd
bash run_bcd.sh whucd
bash run_bcd.sh sysucd
```

The wrapper uses upstream BCD defaults: 256x256 inputs, batch size 16, 80,000 iterations, Adam, learning rate `2e-4`, polynomial schedule, and the upstream augmentation pipeline. Results are written under `Change3D_work_dirs/<dataset>/`; the trainer adds its dataset/iteration/rate subdirectory.

Resource-only values can be overridden, for example:

```bash
GPU_ID=1 BATCH_SIZE=16 NUM_WORKERS=8 bash run_bcd.sh levircd
```

The repository has no standalone BCD test command. Its training script creates a test loader and uses it for validation every epoch, selects `best_model.pth` by that score, then evaluates the same test loader again at the end. It constructs a validation loader but does not use it. Running the unchanged script is useful for checking execution, but its reported test score is not an unbiased held-out result. This adapter leaves that evaluation logic untouched; do not use those scores as final benchmark results without deciding how to handle this protocol issue.

The trainer catches pretrained-weight loading errors and continues with a randomly initialized X3D. Check for `Load pretrained weight` in startup output; `Failed to load pretrained weights` means the intended initialization was not used.
