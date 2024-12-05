num_gpus=1
per_gpu_batchsize=8


# === EHR-XQA ===
# '''
# python main_t5_m3ae.py with data_root=data/finetune_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_vqa_ehr_xqa \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=downloaded/roberta-base \
#  load_path=downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_77.4.ckpt

# '''
# === VQA-RAD ===
python main_t5_m3ae.py with data_root=data/finetune_arrows/ \
 num_workers=0 \
 max_epoch=70 \
 learning_rate=0.00001 \
 batch_size=64 \
 unfreeze_num_encoder_layers=2 \
 unfreeze_num_decoder_layers=2 \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 tokenizer=downloaded/roberta-base \
 load_path=downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_77.4.ckpt \
 mm_encoder_inputs_include_cls_feats=True \
 mm_encoder_inputs_include_imagetext_feats=False \
 mm_encoder_inputs_mm_feats_width=0 \

# python main_t5_m3ae.py with data_root=data/finetune_arrows/ \
#  num_workers=0 \
#  max_epoch=70 \
#  learning_rate=0.00001 \
#  batch_size=64 \
#  unfreeze_num_encoder_layers=3 \
#  unfreeze_num_decoder_layers=3 \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_vqa_vqa_rad \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=downloaded/roberta-base \
#  load_path=downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_77.4.ckpt 


# python main.py with data_root=data/finetune_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_vqa_medvqa_2019 \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=downloaded/roberta-base \
#  load_path=downloaded/pretrained/m3ae.ckpt \
#  clip_resizedcrop

# # === CLS ===
# python main.py with data_root=data/finetune_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_cls_melinda_p_meth_label \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=downloaded/roberta-base \
#  load_path=downloaded/pretrained/m3ae.ckpt \
#  clip_resizedcrop

# # === IRTR ===
# python main.py with data_root=data/finetune_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_irtr_roco get_recall_metric=True \
#  pwdper_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=288 \
#  tokenizer=downloaded/roberta-base \
#  test_only=True \
#  load_path=downloaded/pretrained/m3ae.ckpt

# python main.py with data_root=data/finetune_arrows/ \
#  num_gpus=${num_gpus} num_nodes=1 \
#  task_finetune_irtr_roco get_recall_metric=False \
#  per_gpu_batchsize=${per_gpu_batchsize} \
#  clip16 text_roberta \
#  image_size=384 \
#  tokenizer=downloaded/roberta-base \
#  load_path=downloaded/pretrained/m3ae.ckpt \
#  clip_resizedcrop