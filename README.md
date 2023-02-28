Working Cog-ified AITemplate SD 1.5 inference. 

To build - 
- cog run python ait_sd/scripts/download_pipeline.py (optional args --pipeline "some_org/some_stable_diffusion_pipeline")
- cog run python ait_sd/scripts/compile.py (optional args --width 768 --height 768 --batch_size 4)
- cog predict -i prompt="a cool dog, trending on artstation" 
NOTE - right now ait_sd/scripts/predict.py is configured for two pipelines; pretty straightforward to get it back to 1. 

To push to replicate:
- to reduce space, delete weights/diffusers-pipeline/[unet, vae, text_encoder] (unneeded)
- cog push r8.im/username/modelname
