import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel
from torch.nn import functional as F

from m3ae.modules import M3AETransformerSS
from m3ae.modules import m3ae_t5_utils
from m3ae.modules import objectives

# def PadMask(padded_input, input_lengths=None, pad_idx=None):
#     """ Create a mask to identify non-padding positions.

#     Args:
#         padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
#         input_lengths: Optional, the actual lengths of each sequence before padding, shape (N,).
#         pad_idx: Optional, the index used for padding tokens.

#     Returns:
#         A mask tensor with shape (N, T), where padding positions are marked with 1 and non-padding positions are marked with 0.
#     """

#     # If input is a 2D tensor (N, T), add an extra dimension
#     if padded_input.dim() == 2:
#         padded_input = padded_input.unsqueeze(-1)

#     N, T, _ = padded_input.shape
#     mask = torch.ones((N,T), dtype=torch.bool)

#     if input_lengths is not None:
#         for i in range(N):
#             mask[i, :input_lengths[i]] = False
#     else:
#         # TODO: Infer the mask from the padding index.
#         mask = (padded_input.squeeze(-1) == pad_idx)  # Shape (N, T)

#     return mask


def CausalMask(input_tensor):
    """
    Create an attention mask for causal self-attention based on input lengths.

    Args:
        input_tensor (torch.Tensor): The input tensor of shape (N, T, *).

    Returns:
        attn_mask (torch.Tensor): The causal self-attention mask of shape (T, T)
    """
    T = input_tensor.shape[1]  # Sequence length
    # TODO: Initialize attn_mask as a tensor of zeros with the right shape.
    attn_mask = torch.zeros((T,T), dtype=torch.bool) # Shape (T, T)

    # TODO: Create a lower triangular matrix to form the causal mask.
    causal_mask = ~torch.tril(torch.ones((T,T), dtype=torch.bool), diagonal=0) # Lower triangular matrix

    # TODO: Combine the initial mask with the causal mask.
    attn_mask = attn_mask | causal_mask

    return attn_mask

class PositionalEncoding(torch.nn.Module):
    ''' Position Encoding from Attention Is All You Need Paper '''

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        # self.d_model = d_model
        # self.max_len = max_len
        # Initialize a tensor to hold the positional encodings
        pe          = torch.zeros(max_len, d_model)

        # Create a tensor representing the positions (0 to max_len-1)
        position    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term for the sine and cosine functions
        # This term creates a series of values that decrease geometrically, used to generate varying frequencies for positional encodings
        div_term    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the positional encodings tensor and make it a buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
      return x + self.pe[:, :x.size(1)]

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.mha1       = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mha2       = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn        = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.identity   = nn.Identity()
        self.pre_norm   = nn.LayerNorm(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.dropout3   = nn.Dropout(dropout)

    def forward(self, padded_targets, enc_output, pad_mask_dec, slf_attn_mask):

        #   Step 1: Self Attention
        #   (1) pass through the Multi-Head Attention (Hint: you need to store weights here as part of the return value)
        #   (2) add dropout
        #   (3) residual connections
        #   (4) layer normalization

        # print(f"padded target size in decoder layer: {padded_targets.size()}")
        residual = padded_targets

        x_norm = self.pre_norm(padded_targets)

        mha1_output, mha1_attn_weights = self.mha1(x_norm,x_norm,x_norm,attn_mask=slf_attn_mask,
                                                   key_padding_mask=pad_mask_dec)
        x = residual + self.dropout1(mha1_output)
        residual = x
        x = self.layernorm1(x)


        #   Step 2: Cross Attention
        #   (1) pass through the Multi-Head Attention (Hint: you need to store weights here as part of the return value)
              #  think about if key,value,query here are the same as the previous one?
        #   (2) add dropout
        #   (3) residual connections
        #   (4) layer normalization

        if enc_output is None:
            mha2_output       = self.identity(padded_targets)
            mha2_attn_weights = torch.zeros_like(mha1_attn_weights)
        else:
            enc_pad_mask = None
            if enc_output is not None:
                enc_pad_mask = torch.zeros(
                    enc_output.size(0), enc_output.size(1), 
                    dtype=torch.bool, 
                    device=enc_output.device
                )

            mha2_output, mha2_attn_weights = self.mha2(x, enc_output, enc_output,
                                          key_padding_mask=enc_pad_mask)
        x = self.dropout2(mha2_output)
        x = x + residual
        residual = x
        x = self.layernorm2(x)


        #   Step 3: Feed Forward Network
        #   (1) pass through the FFN
        #   (2) add dropout
        #   (3) residual connections
        #   (4) layer normalization
        ffn_output = self.ffn(x)
        x = self.dropout3(ffn_output)
        x = x + residual
        x = self.layernorm3(x)

        # return the output of the decoder layer
        return x, mha1_attn_weights, mha2_attn_weights


class Decoder(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff, dropout,
                 max_len,
                 target_vocab_size):

        super().__init__()

        self.max_len        = max_len
        self.num_layers     = num_layers
        self.num_heads      = num_heads

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.target_embedding       = nn.Embedding(target_vocab_size, d_model)  # use torch.nn.Embedding
        self.positional_encoding    = PositionalEncoding(d_model) #max_len
        self.final_linear           = nn.Linear(d_model, target_vocab_size)
        self.dropout                = nn.Dropout(dropout)


    def forward(self, padded_targets, padding_mask, cross_attn_feats):

        pad_mask_dec = ~padding_mask
        causal_mask = CausalMask(input_tensor=padded_targets).to(padded_targets.device)

        # Step1:  Apply the embedding
        target_embed = self.target_embedding(padded_targets)

        # Step2:  Apply positional encoding
        target_embed += self.positional_encoding(target_embed)
        target_embed = self.dropout(target_embed)

        # Step4: Pass through decoder layers
        runnint_att = {}
        for i in range(self.num_layers):
            x, runnint_att['layer{}_dec_self'.format(i + 1)], runnint_att['layer{}_dec_cross'.format(i + 1)] = self.dec_layers[i](
                target_embed, cross_attn_feats, pad_mask_dec, causal_mask
                )

        # Step5: linear layer (Final Projection) for next character prediction
        seq_out = self.final_linear(x)

        return seq_out, runnint_att


    def recognize_greedy_search(self, cross_attn_feats, tokenizer):
        ''' Greedy search generation with multi-modal features '''
        batch_size = cross_attn_feats.size(0)
        target_seq = torch.full((batch_size, 1), tokenizer.cls_token_id, dtype=torch.long).to(cross_attn_feats.device)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(cross_attn_feats.device)
        
        # Explicitly get the [SEP] token ID
        sep_token_id = tokenizer.sep_token_id
        
        for step in range(self.max_len):
            seq_out, running_att = self.forward(target_seq, None, cross_attn_feats)
            logits = torch.nn.functional.log_softmax(seq_out[:, -1], dim=1)

            # Handle special token conditions more explicitly
            next_token = logits.argmax(dim=-1).unsqueeze(1)
            
            # Check if the next token is [SEP] or [EOS]
            sep_eos_mask = (next_token.squeeze(-1) == sep_token_id) | \
                        (next_token.squeeze(-1) == tokenizer.eos_token_id)
            
            # Update finished status
            finished |= sep_eos_mask
            
            # Append next token
            target_seq = torch.cat([target_seq, next_token], dim=-1)
            
            # Early stopping if all sequences are finished
            if finished.all():
                break
        
        # Remove the initial cls token
        target_seq = target_seq[:, 1:]
        
        # Truncate sequences at first [SEP] or [EOS]
        for i in range(batch_size):
            # Find the first occurrence of [SEP] or [EOS]
            special_token_indices = torch.where(
                (target_seq[i] == sep_token_id) | 
                (target_seq[i] == tokenizer.eos_token_id)
            )[0]
            
            if len(special_token_indices) > 0:
                first_special_index = special_token_indices[0]
                target_seq[i, first_special_index+1:] = tokenizer.pad_token_id
        
        # Pad sequences to max length
        max_length = target_seq.size(1)
        target_seq = torch.nn.functional.pad(
            target_seq, 
            (0, self.max_len - max_length), 
            value=tokenizer.pad_token_id
        )

        return target_seq


class DecoderModel(pl.LightningModule):
    def __init__(self, m3ae_config, max_answer_length=12, min_answer_length=1, freeze_m3ae=True):
        super().__init__()
        self.save_hyperparameters()

        # Init tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Initialize M3AE model
        self.m3ae = M3AETransformerSS(m3ae_config)

        self.decoder = Decoder(
            num_layers=6,
            d_model=768,
            num_heads=8,
            d_ff=768*4,
            dropout=0.1,
            max_len=128,
            target_vocab_size=self.tokenizer.vocab_size
        )

        # Freeze M3AE if specified
        if freeze_m3ae:
            for param in self.m3ae.parameters():
                param.requires_grad = False
        

        self.max_answer_length = max_answer_length
        self.min_answer_length = min_answer_length
        m3ae_t5_utils.set_metrics(self)
        self.current_tasks = list()

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    


    def forward(self, batch, test=False):
        # Prepare encoder inputs with combined visual and textual features


        # Extract multi-modal features from M3AE
        m3ae_output = self.m3ae.infer(batch, mask_text=False, mask_image=False)

        multi_modal_cls_feats = torch.cat([
            m3ae_output["multi_modal_image_feats"], 
            m3ae_output["multi_modal_text_feats"]
        ], dim=1)

        # print(f"size of mm cls feats: {multi_modal_cls_feats.size()}")

        ret = dict()


        if len(self.current_tasks) == 0 or test:
            # Inference/test mode - generate text
            generated_texts = self.decoder.recognize_greedy_search(
                multi_modal_cls_feats, 
                tokenizer=self.tokenizer
            )
            
            generated_texts_ = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_texts]
            ret.update({'outputs': generated_texts_})
            return ret

        else:
            # Training mode
            targets = batch["vqa_answer"]
            targets = [ans[0] for ans in targets]
            # print(f"\n training targets {targets}")
            
            # Tokenize and pad targets
            target_tokens = self.tokenizer(
                targets, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=self.max_answer_length
            ).input_ids.to(self.device)

            # remove start of sequence token
            target_tokens = target_tokens[:, 1:]
            
            padding_mask = torch.ne(target_tokens, self.tokenizer.pad_token_id)
            # print(f"\n training targets {targets}")
            # print(f"\n training target tokens: {target_tokens}")

            # Get model outputs
            output, attention_weights = self.decoder(
                target_tokens, 
                padding_mask, 
                multi_modal_cls_feats
            )
            
            # Compute loss with proper padding mask
            # print(f"\n model outputs: {output}")
            
            ce_loss = self.criterion(output.transpose(1, 2), target_tokens) * padding_mask
            loss = ce_loss.sum() / padding_mask.sum()
            
            # Decode generated text for metrics
            with torch.no_grad():
                generated_texts = [
                    self.tokenizer.decode(seq, skip_special_tokens=True) 
                    for seq in torch.argmax(output, dim=-1)
                ]
                generated_texts_ = [[item] for item in generated_texts]
            
            ret.update(objectives.compute_vqa(
                self, 
                batch, 
                test=test, 
                outputs=generated_texts_,
                loss=loss,
                labels=targets
            ))
            
            return ret


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