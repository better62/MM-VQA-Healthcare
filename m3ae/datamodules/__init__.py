from .irtr_roco_datamodule import IRTRROCODataModule
from .pretraining_medicat_datamodule import MedicatDataModule
from .pretraining_roco_datamodule import ROCODataModule
from .vqa_vqa_rad_datamodule import VQAVQARADDataModule
from .vqa_ehr_xqa_datamodule import VQAEHRXQADataModule

_datamodules = {
    "medicat": MedicatDataModule,
    "roco": ROCODataModule,
    "vqa_ehr_xqa":VQAEHRXQADataModule,
    "vqa_vqa_rad": VQAVQARADDataModule,
    "irtr_roco": IRTRROCODataModule
}
