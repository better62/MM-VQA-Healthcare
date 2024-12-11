num_gpus=1
per_gpu_batchsize=32


# '''
# === VQA-RAD ===
# ==================== DECODER ONLY ABLATIONS ==============================
# python3.8 main_decoder_m3ae.py with data_root=data/finetune_arrows_vqa_rad/ \
#  num_workers=8 \
#  max_epoch=30 \
#  t5_max_length=50 \
#  learning_rate=0.0001 \
#  batch_size=64 \
#  mm_encoder_inputs_include_cls_feats=False \
#  mm_encoder_inputs_include_imagetext_feats=True \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_vqa_vqa_rad \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=downloaded/roberta-base \
#  load_path=downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_77.4.ckpt 

# python3.8 main_decoder_m3ae.py with data_root=data/finetune_arrows_vqa_rad/ \
#  num_workers=8 \
#  max_epoch=20 \
#  t5_max_length=50 \
#  learning_rate=0.0001 \
#  batch_size=64 \
#  mm_encoder_inputs_include_cls_feats=True \
#  mm_encoder_inputs_include_imagetext_feats=False \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_vqa_vqa_rad \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=downloaded/roberta-base \
#  load_path=downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_77.4.ckpt \
#  decoder_load_path=downloaded/finetuned/vqa/vqa_rad/decoder_clsonly_16.ckpt

python3.8 main_decoder_m3ae.py with data_root=data/finetune_arrows_vqa_rad/ \
 num_workers=8 \
 max_epoch=15 \
 t5_max_length=50 \
 learning_rate=0.0001 \
 batch_size=64 \
 mm_encoder_inputs_include_cls_feats=True \
 mm_encoder_inputs_include_imagetext_feats=True \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 load_path=downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_77.4.ckpt \
 decoder_load_path=downloaded/finetuned/vqa/vqa_rad/decoder_cls_imgtext_20.ckpt
 # ==================== DECODER ONLY ABLATIONS ==============================
