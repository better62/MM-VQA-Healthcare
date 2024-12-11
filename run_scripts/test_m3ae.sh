num_gpus=1
per_gpu_batchsize=16

python3.8 main.py with data_root=data/finetune_arrows_m3ae/ \
 num_workers=0 \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 test_only=True \
 tokenizer=downloaded/roberta-base \
 load_path=downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad.ckpt \


