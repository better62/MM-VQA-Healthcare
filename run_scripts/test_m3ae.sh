num_gpus=1
per_gpu_batchsize=16

# === 1. VQA ===
# === VQA-RAD ===
# === Provided Checkpoints ===
# downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_77.4.ckpt
# downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_76.9.ckpt
# downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_76.7.ckpt

python3.8 main.py with data_root=data/finetune_arrows_m3ae/ \
 num_workers=0 \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 test_only=True \
 tokenizer=downloaded/roberta-base \
 load_path=checkpoints/task_finetune_vqa_vqa_rad-seed0-from_downloaded_finetuned_vqa_vqa_rad_m3ae_finetuned_vqa_vqa_rad_77.4.ckpt_VQA-RAD-T5/26_lc3ptblt/checkpoints/last.ckpt
 #above load_path should be changed to the path of the ckpt that you want to use