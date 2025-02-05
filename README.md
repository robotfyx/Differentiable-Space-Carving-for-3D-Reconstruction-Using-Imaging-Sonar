# Differentiable Space Carving Code

## Preparatory Work

1. Dataset: We have put the datasets collected in the two simulation environments in the folder `data`. Please put the new dataset in this folder as well.
2. Weights used for denoising: Download the pretrained weights `scunet_gray_25.pth` to the folder denoise according to the tutorial in [SCUNet](
https://github.com/cszn/SCUNet/blob/main/main_download_pretrained_models.py).

## Train&Reconstrution

run

`python run.py --conf 14deg_simpleBoat --name 14deg_noise_simpleBoat --expname test --iters 3`in the teminal to start training.
After training, you can get the training results and the reconstructed .ply model in folder `results/test`















