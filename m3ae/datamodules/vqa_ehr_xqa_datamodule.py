from collections import defaultdict

from .base_datamodule import BaseDataModule
from ..datasets import VQAEHRXQADataset


class VQAEHRXQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VQAEHRXQADataset

    @property
    def dataset_name(self):
        return "vqa_ehr_xqa"

    def setup(self, stage):
        super().setup(stage)
      
