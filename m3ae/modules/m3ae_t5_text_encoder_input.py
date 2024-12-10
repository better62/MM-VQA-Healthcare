import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import functional as F

from m3ae.modules import M3AETransformerSS
from m3ae.modules import m3ae_t5_utils
from m3ae.modules import objectives

class T5VQA_TextEncoderInput(pl.LightningModule):
    def __init__(self, m3ae_config, max_answer_length=12, min_answer_length=1, freeze_m3ae=True, freeze_t5_layers=True):
        super().__init__()
        self.save_hyperparameters()

        # Initialize M3AE model
        self.m3ae = M3AETransformerSS(m3ae_config)

        # Freeze M3AE if specified
        if freeze_m3ae:
            for param in self.m3ae.parameters():
                param.requires_grad = False
        
        # Initialize T5 with generation capabilities
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-small')

        # Freeze T5 layers if specified
        if freeze_t5_layers:
            for param in self.t5.parameters():
                param.requires_grad = False
        
        # Projection layer for M3AE features
        self.feature_projection = nn.Linear(
            m3ae_config["hidden_size"] * 2,  # M3AE outputs concatenated features
            self.t5.config.hidden_size
        )
        
        self.max_answer_length = max_answer_length
        self.min_answer_length = min_answer_length
        m3ae_t5_utils.set_metrics(self)
        self.current_tasks = list()

        # Load checkpoint if path is provided
        if m3ae_config["load_path"] != "":
            self.load_model_checkpoint(m3ae_config["load_path"])        

    def load_model_checkpoint(self, ckpt_path):
        # Load the checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        
        # Load M3AE weights with positional encoding adjustment if needed
        if "m3ae" in state_dict:
            if self.m3ae.is_clip:
                state_dict = adapt_position_encoding(state_dict,
                                                     after=self.m3ae.hparams.config["image_size"],
                                                     patch_size=self.m3ae.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict,
                                                          after=self.m3ae.hparams.config["image_size"])
            self.m3ae.load_state_dict(state_dict, strict=False)
        
        # Load T5 weights
        t5_state_dict = {k[len("t5."):]: v for k, v in state_dict.items() if k.startswith("t5.")}
        self.t5.load_state_dict(t5_state_dict, strict=False)

        print("Checkpoint loaded successfully!")
            

    
    def projection_layer(self, input_dim, output_dim=512):
        linear_layer = nn.Linear(input_dim, output_dim).to(self.device)
        return linear_layer

    def norm_layer(self, dim=512):
        norm_layer = nn.LayerNorm(dim).to(self.device)
        return norm_layer
    
    def unfreeze_top_layers(self, num_encoder_layers=2, num_decoder_layers=2):
        # Freeze all T5 layers by default
        for param in self.t5.parameters():
            param.requires_grad = False

        # Unfreeze top N encoder layers
        for i in range(len(self.t5.encoder.block) - num_encoder_layers, len(self.t5.encoder.block)):
            for param in self.t5.encoder.block[i].parameters():
                param.requires_grad = True

        # Unfreeze top N decoder layers
        for i in range(len(self.t5.decoder.block) - num_decoder_layers, len(self.t5.decoder.block)):
            # Self attention
            for param in self.t5.decoder.block[i].layer[0].parameters():
                param.requires_grad = True
            # Cross attention
            for param in self.t5.decoder.block[i].layer[1].parameters():
                param.requires_grad = True
        
        #print(f"Unfroze {num_encoder_layers} encoder layers and {num_decoder_layers} decoder layers.")

    def prepare_inputs(self, batch):
        
        question_tokens = self.tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            max_length=self.hparams.m3ae_config["max_text_len"]
        ).input_ids.to(self.device)

        question_embeddings = self.t5.shared(question_tokens)  # 1 x question_len x 512

        # Extract multi-modal features from M3AE
        m3ae_output = self.m3ae.infer(batch, mask_text=False, mask_image=False)
        multi_modal_cls_feats = m3ae_output["multi_modal_cls_feats"]

        # Project and normalize MM features
        cls_proj_layer = self.projection_layer(multi_modal_cls_feats.shape[1])
        projected_cls_feature = cls_proj_layer(cls_feature)  # [batch size, 512]

        cls_norm_layer = self.norm_layer()
        norm_cls_feature = cls_norm_layer(projected_cls_feature)

        # Repeat multi-modal features to match question embedding length
        repeated_cls_feature = norm_cls_feature.unsqueeze(1).repeat(1, question_embeddings.shape[1], 1)
     
        combined_features = torch.cat([
            question_embeddings,
            repeated_cls_feature
        ], dim=-1)

        fusion_layer = self.projection_layer(self.t5.config.hidden_size * 2, self.t5.config.hidden_size)

        fused_features = fusion_layer(combined_features)
      
        return {
            "inputs_embeds": fused_features,
            "attention_mask": torch.ones_like(question_tokens, dtype=torch.long)
        }


    def forward(self, batch, test=False):
        # Prepare encoder inputs with combined visual and textual features
        ret = dict()
        
        model_inputs = self.prepare_inputs(batch)
    
        encoder_outputs = self.t5.encoder(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"]
        )

        if len(self.current_tasks) == 0 or test:
            # Inference/test mode - generate text
            outputs = self.t5.generate(
                encoder_outputs=encoder_outputs,
                max_length=self.max_answer_length,
                min_length=self.min_answer_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Decode the generated outputs to text
            generated_texts = self.tokenizer.batch_decode(
                outputs.sequences, 
                skip_special_tokens=True
            )
            generated_texts_ = [[item] for item in generated_texts]
            ret.update({'outputs': generated_texts_})
            return ret

        else:
            # Training mode
            labels = batch["vqa_answer"]  # Assuming these are text answers
            flattened_labels = [label[0] for label in labels]
            # Tokenize the labels
            label_tokens = self.tokenizer(
                flattened_labels,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Get model outputs
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=label_tokens,
                return_dict=True
            )
            
            # For metrics, we need to generate text even during training
            with torch.no_grad():
                generated_outputs = self.t5.generate(
                    encoder_outputs=encoder_outputs,
                    max_length=self.max_answer_length,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
                
                # Decode generated text
                generated_texts = self.tokenizer.batch_decode(
                    generated_outputs.sequences, 
                    skip_special_tokens=True
                )

                generated_texts_ = [[item] for item in generated_texts]
            
            ret.update(objectives.compute_vqa(
                self, 
                batch, 
                test=test, 
                outputs=generated_texts_,  # Pass decoded text
                loss=outputs.loss,
                labels=labels  # Pass original text labels
            ))
            
            return ret

    def generate_answer(self, batch):
        outputs = self(batch)
        
        # Decode generated sequences
        answers = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return answers

    def training_step(self, batch, batch_idx):
        m3ae_t5_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v * self.hparams.m3ae_config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])

        return {'loss': total_loss}

    def training_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_t5_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_t5_utils.set_task(self)
        output = self(batch, test=True)

    def test_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self, test=True)

    def training_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_t5_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_t5_utils.set_task(self)
        output = self(batch, test=True)

    def test_epoch_end(self, outs):
        m3ae_t5_utils.epoch_wrapup(self, test=True)


    def configure_optimizers(self):
        return m3ae_t5_utils.set_schedule(self)