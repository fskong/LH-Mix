from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
from transformers.modeling_outputs import (
    MaskedLMOutput
)
from transformers.activations import ACT2FN
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import os
from .loss import multilabel_categorical_crossentropy
import math
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np



class UpdateEmbedding(nn.Module):
    def __init__(self, config, embedding, new_embedding, layer=1, path_list=None, data_path=None):
        super(UpdateEmbedding, self).__init__()
        padding_idx = config.pad_token_id
        self.num_class = config.num_labels
        self.padding_idx = padding_idx
        self.original_embedding = embedding
        new_embedding = torch.cat([torch.zeros(1, new_embedding.size(-1), device=new_embedding.device, dtype=new_embedding.dtype),
                                    new_embedding], dim=0)
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, False, 0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        self.depth = (self.new_embedding.num_embeddings - 2 - self.num_class)

    @property
    def raw_weight(self):
        def foo():
            return torch.cat([self.original_embedding.weight, self.new_embedding.weight[1:, :]], dim=0)   # from 1, delete the cat zeros, may be root

        return foo

    def forward(self, x):
        x = F.embedding(x, self.raw_weight(), self.padding_idx)   # F.embedding, select embedding by index
        return x


class OutputEmbedding(nn.Module):
    def __init__(self, bias):
        super(OutputEmbedding, self).__init__()
        self.weight = None
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


class Prompt(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, layer=1, path_list=None, data_path=None, depth2label=None, mix_alpha=None, warm_up=None, threshold=None, alpha=None, **kwargs):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.cls = BertOnlyMLMHead(config)
        self.num_labels = config.num_labels
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)
        self.data_path = data_path
        self.vocab_size = self.tokenizer.vocab_size
        print ('__init__ self.vocab_size', self.vocab_size)
        self.path_list = path_list
        self.depth2label = depth2label
        self.layer = layer
        self.init_weights()
        self.mix_alpha = mix_alpha
        self.warm_up = warm_up
        self.threshold = threshold
        self.alpha = alpha

    def get_output_embeddings(self):
        return self.cls.predictions.decoder   # nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_embedding(self):
        depth = len(self.depth2label)
        label_dict = torch.load(os.path.join(self.data_path, 'value_dict.pt'))
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        label_dict = {i: v for i, v in label_dict.items()}
        if 'rcv1' in self.data_path:
            label_des = torch.load(os.path.join(self.data_path, 'label_description.pt'))
            for idx, name in label_dict.items():
                label_dict[idx] = label_des[name.lower()]
        label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()}
        label_emb = []
        input_embeds = self.get_input_embeddings()
        for i in range(len(label_dict)):
            label_emb.append(input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id], device=self.device, dtype=torch.long))
        prompt_embedding = nn.Embedding(depth + 1, input_embeds.weight.size(1), 0)

        self._init_weights(prompt_embedding)
        label_emb = torch.cat([torch.stack(label_emb), prompt_embedding.weight[1:, :], prefix], dim=0)
        embedding = UpdateEmbedding(self.config, input_embeds, label_emb,
                                   path_list=self.path_list, layer=self.layer, data_path=self.data_path)

        self.set_input_embeddings(embedding)
        output_embeddings = OutputEmbedding(self.get_output_embeddings().bias)
        self.set_output_embeddings(output_embeddings)
        output_embeddings.weight = embedding.raw_weight
        self.vocab_size = output_embeddings.bias.size(0)
        output_embeddings.bias.data = nn.functional.pad(
            output_embeddings.bias.data,
            (
                0,
                embedding.size - output_embeddings.bias.shape[0],
            ),
            "constant",
            0,
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            all_label_flat=None,
            epoch=0,

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1) 
        single_labels = input_ids.masked_fill(multiclass_pos | (input_ids == self.config.pad_token_id), -100)
        mix_lambda = 1
        mix_index = np.asarray(range(input_ids.size(0)))

        if self.training:
            enable_mask = input_ids < self.tokenizer.vocab_size
            random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
            input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
            random_ids = torch.randint_like(input_ids, 104, self.vocab_size)
            mlm_mask = random_mask > 0.985
            input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask
            mlm_mask = random_mask < 0.85
            single_labels = single_labels.masked_fill(mlm_mask, -100)

            if epoch >= self.warm_up:
                mix_lambda = np.random.beta(self.mix_alpha, self.mix_alpha)
                mix_lambda = max(mix_lambda, 1 - mix_lambda)
                mix_index = torch.randperm(input_ids.size(0))
                label_out = self.bert(all_label_flat)[0][:, 0]
                cosine_similarity = torch.cosine_similarity(label_out, label_out[mix_index], dim=1)
                cosine_similarity = 0.5 * (cosine_similarity + 1)
                mix_lambda = -(self.threshold - 0.5) * torch.pow(cosine_similarity, self.alpha) + self.threshold
                mix_lambda = mix_lambda.unsqueeze(-1).unsqueeze(-1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        prediction_scores_mix = self.cls(sequence_output * mix_lambda + sequence_output[mix_index] * (1 - mix_lambda))
        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), single_labels.view(-1))
            multiclass_logits = prediction_scores_mix.masked_select(
                multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores_mix.size(-1))).view(-1, prediction_scores_mix.size(-1))
            
            multiclass_logits = multiclass_logits[:, self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias

            labels = labels.view(-1, self.num_labels)
            mix_index_bydepth = torch.range(0, len(self.depth2label) * input_ids.size(0) - 1).view(-1, len(self.depth2label))
            mix_index_bydepth = mix_index_bydepth[mix_index].view(-1).long()

            multiclass_loss = multilabel_categorical_crossentropy(labels, multiclass_logits, mix_lambda) \
                            + multilabel_categorical_crossentropy(labels[mix_index_bydepth], multiclass_logits, (1 - mix_lambda))
            masked_lm_loss += multiclass_loss

        if not return_dict:
            output = (prediction_scores_mix,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        ret = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores_mix,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return ret

    @torch.no_grad()
    def generate(self, input_ids, depth2label, **kwargs):
        attention_mask = input_ids != self.config.pad_token_id
        outputs = self(input_ids, attention_mask)
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        prediction_scores = outputs['logits']
        prediction_scores = prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1, prediction_scores.size(-1))
        
        prediction_scores = prediction_scores[:, self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias    # torch.Size([depth*16, label_num])
        prediction_scores = prediction_scores.view(-1, len(depth2label), prediction_scores.size(-1))
        predict_labels = []
        for scores in prediction_scores:
            predict_labels.append([])
            for i, score in enumerate(scores):
                for l in depth2label[i]:
                    if score[l] > 0:
                        predict_labels[-1].append(l)
        return predict_labels, prediction_scores
