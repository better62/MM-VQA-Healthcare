import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "M3AE"))
sys.path.append(os.path.join(os.getcwd(), "M3AE/prepro"))
print(sys.path)

from prepro.prepro_finetuning_data import prepro_vqa_vqa_rad

data_root = os.path.join(os.getcwd(), "M3AE/data/finetune_data/vqa_rad")
print(data_root)
prepro_vqa_vqa_rad(data_root)