## Medical Visual Question Answering: Improving Answer Quality on Multimodal Datasets with a Generative Output Layer
Healthcare data is inherently multimodal, combining textual (e.g., clinical notes) and visual (e.g., medical images) components. Medical Visual Question Answering (Med-VQA) seeks to address this by enabling AI models to interpret multimodal data and answer clinically relevant questions. However, most Med-VQA models rely on classification-based approaches, which constrain adaptability and clinical applicability due to the use of a fixed answer set.

We propose reframing Med-VQA as a generative task by replacing the classification head in the state-of-the-art M$^3$AE model with a generative layer. This approach allows the model to produce open-ended, contextually relevant responses, improving its flexibility and practical utility. Experimental results demonstrate significant enhancements in question-answering performance, highlighting the potential of generative Med-VQA models to support diverse and real-world clinical applications effectively.


## Requirements
Run the following command to install the required packages. Keep in mind this will require Python 3.8.
```bash
pip install -r requirements.txt
```

## Weights
Weights are available [here]()

## Downstream Evaluation
### 1. Dataset Preparation
Please organize the fine-tuning datasets as the following structure:
```angular2
root:[data]
+--finetune_data
| +--vqa_rad
| | +--trainset.json
| | +--valset.json
| | +--testset.json
| | +--images
```

### 2. Pre-processing
Run the following command to pre-process the data:
```angular2
python prepro/prepro_finetuning_data.py
```
to get the following arrow files:
```angular2
root:[data]
+--finetune_arrows
| +--vqa_vqa_rad_train.arrow
| +--vqa_vqa_rad_val.arrow
| +--vqa_vqa_rad_test.arrow
```

### 3. Fine-Tuning:
```angular2
bash run_scripts/finetune_m3ae.sh
```

### 4. Test:
```angular2
bash run_scripts/test_m3ae.sh
```

