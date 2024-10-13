## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
DATA_ROOT=("/content/drive/MyDrive/DistilDire/dataset/dalle3_males" "/content/drive/MyDrive/DistilDire/dataset/dalle3_females" "/content/drive/MyDrive/DistilDire/dataset/midjourney_males" "/content/drive/MyDrive/DistilDire/dataset/midjourney_females" "/content/drive/MyDrive/DistilDire/dataset/sd1_males" "/content/drive/MyDrive/DistilDire/dataset/sd1_females")
SAVE_ROOT=("/content/drive/MyDrive/DistilDire/dataset/dalle3_males" "/content/drive/MyDrive/DistilDire/dataset/dalle3_females" "/content/drive/MyDrive/DistilDire/dataset/midjourney_males" "/content/drive/MyDrive/DistilDire/dataset/midjourney_females" "/content/drive/MyDrive/DistilDire/dataset/sd1_males" "/content/drive/MyDrive/DistilDire/dataset/sd1_females")

MODEL_PATH="/content/drive/MyDrive/DistilDire/AdmModel/256x256_diffusion_uncond.pt" # imagenet pretrained adm (unconditional, 512x512)
SAMPLE_FLAGS="--batch_size 32" # ddim20 is forced
PREPROCESS_FLAGS="--compute_dire True --compute_eps True"

for i in 0 1 2 3 4 5
do
    SAVE_FLAGS="--data_root ${DATA_ROOT[$i]} --save_root ${SAVE_ROOT[$i]}"
    echo "Running on ${DATA_ROOT[$i]} with save root ${SAVE_ROOT[$i]}"
    torchrun --standalone --nproc_per_node 1 -m guided_diffusion.compute_dire_eps --model_path $MODEL_PATH $PREPROCESS_FLAGS $SAMPLE_FLAGS $SAVE_FLAGS
done
# torchrun --standalone --nproc_per_node 8 -m train --batch 128 --exp_name tm-global-scale --datasets y1-global-truemedia --epoch 40 --lr 1e-4