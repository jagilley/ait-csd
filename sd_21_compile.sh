#!/bin/bash

python ait_sd/scripts/download_pipeline.py --save_directory ./weights_sd21/diffusers-pipeline/

python ait_sd/scripts/compile.py --local-dir ./weights_sd21/diffusers-pipeline/ --compile_out ./weights_sd21_512 

python ait_sd/scripts/compile.py --local-dir ./weights_sd21/diffusers-pipeline/ --compile_out ./weights_sd21_768 --height 768 --width 768

python ait_sd/scripts/demo.py --local-dir ./weights_sd21/diffusers-pipeline/ --compile-dir ./weights_sd21_768

python ait_sd/scripts/demo.py --local-dir ./weights_sd21/diffusers-pipeline/ --compile-dir ./weights_sd21_512

rm -rf weights_sd21/diffusers-pipeline/unet/
rm -rf weights_sd21/diffusers-pipeline/vae/
rm -rf weights_sd21/diffusers-pipeline/text_encoder/