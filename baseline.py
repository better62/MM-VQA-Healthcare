import sys
import os

sys.path.append("/home/vdurham/IDL-Project/MM-T-VQA-Healthcare")
sys.path.append("/home/vdurham/IDL-Project/MM-T-VQA-Healthcare/M3AE")
sys.path.append("/home/vdurham/IDL-Project/MM-T-VQA-Healthcare/M3AE/prepro")
sys.path.append("/home/vdurham/IDL-Project/MM-T-VQA-Healthcare/M3AE/data")

from prepro.prepro_finetuning_data import prepro_vqa_vqa_rad

data_root = os.path.join(os.getcwd(), "M3AE/data/finetune_data/vqa_rad")
print(data_root)
prepro_vqa_vqa_rad(data_root)