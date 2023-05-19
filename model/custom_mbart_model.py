from abc import ABC
from typing import Optional, Tuple, Union
import torch
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput
from transformers.models.mbart import MBartForConditionalGeneration, MBartConfig, MBartModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.mbart.modeling_mbart import (
    MBART_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
    MBART_GENERATION_EXAMPLE,
    shift_tokens_right
)
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from torch.nn import CrossEntropyLoss
import numpy as np
from torch.autograd import Variable
import random



def create_forward(self):
    def forward(
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqModelOutput, Tuple[torch.FloatTensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # different to other models, MBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_outputs["last_hidden_state"] = encoder_outputs["last_hidden_state"].detach()

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    return forward

class CustomMbartModel(MBartForConditionalGeneration, ABC):
    def __init__(self, config: MBartConfig):
        super(CustomMbartModel, self).__init__(config=config)
        self.da_method = None
        self.word_dropout_ratio = None
        self.word_replacement_ratio = None
        self.pad_id = config.pad_token_id
        self.bos_id = config.bos_token_id
        self.eos_id = config.eos_token_id
        # self.model.forward = create_forward(self.model)

    def set_augment_config(self, word_dropout_ratio: float = 0, word_replacement_ratio: float = 0, da_method = None):
        self.da_method = da_method
        self.word_dropout_ratio = word_dropout_ratio
        self.word_replacement_ratio = word_replacement_ratio

    def word_dropout(self, batch_):
        if self.word_dropout_ratio == 0:
            return batch_
        batch = torch.clone(batch_)
        for batch_idx in range(len(batch)):
            for token_idx in range(len(batch[batch_idx])):
                if np.random.randn(1)[0] < self.word_dropout_ratio:
                    batch[batch_idx][token_idx] *= 0
        return batch

    def word_replacement(self, batch_ids_):
        if self.word_replacement_ratio == 0:
            return batch_ids_
        batch_ids = torch.clone(batch_ids_)
        embedding_size = len(self.model.shared.weight)
        for batch_idx in range(len(batch_ids)):
            for token_idx in range(len(batch_ids[batch_idx])):
                if token_idx == self.pad_id:
                    break
                if np.random.randn(1)[0] < self.word_replacement_ratio:
                    batch_ids[batch_idx][token_idx] = np.random.randint(0, embedding_size)
        return batch_ids
    
    def switch_out(self, sents, tau, vocab_size, bos_id, eos_id, pad_id):
        """
        Sample a batch of corrupted examples from sents.

        Args:
        sents: Tensor [batch_size, n_steps]. The input sentences.
        tau: Temperature.
        vocab_size: to create valid samples.
        Returns:
        sampled_sents: Tensor [batch_size, n_steps]. The corrupted sentences.

        """
        mask = torch.eq(sents, bos_id) | torch.eq(sents, eos_id) | torch.eq(sents, pad_id)
        mask = mask.data.type('torch.ByteTensor') #converting to byte tensor for masked_fill in built function
        lengths = mask.float().sum(dim=1)
        batch_size, n_steps = sents.size()

        # first, sample the number of words to corrupt for each sentence
        logits = torch.arange(n_steps, dtype=torch.float)
        large_negative = -1e9
        logits = logits.mul_(-1).unsqueeze(0).expand_as(sents).contiguous().masked_fill_(mask, large_negative)
        logits = Variable(logits)
        probs = torch.nn.functional.softmax(logits.mul_(tau), dim=1)
        # probs_float = torch.nn.functional.softmax(logits.mul_(tau), dim=1)
        # probs = probs_float.type(torch.long)

        # finding corrupt sampels (most likely empty or 1 word) leading to zero prob
        for idx,prob in enumerate(probs.data):
            if torch.sum(prob)<= 0 and idx!=0:
                valid_ind = list(set(range(len(probs.data))))- list(set([idx]))
                for i in range(100):
                    new_indx = random.choice(valid_ind)
                    if not torch.sum(probs.data[new_indx])<= 0:
                        probs[idx] = probs[new_indx]
                        break
                    else:
                        pass

        # still num_words probs fails likely due to corrupt input, therefore returning the whole original batch
        try:
            num_words = torch.distributions.Categorical(probs).sample()
        except:
            print ('Returning orignial batch!!!!!!')
            return sents

        corrupt_pos = num_words.data.float().div_(lengths).unsqueeze(1).expand_as(sents).contiguous().masked_fill_(mask, 0)

        corrupt_pos = corrupt_pos.clamp(min=0, max=1)
        corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).byte()
        total_words = int(corrupt_pos.sum())

        # sample the corrupted values, which will be added to sents
        corrupt_val = torch.LongTensor(total_words)
        corrupt_val = corrupt_val.random_(1, vocab_size)
        corrupts = torch.zeros(batch_size, n_steps).long()
        corrupts = corrupts.masked_scatter_(corrupt_pos, corrupt_val)
        # corrupts = corrupts.cuda()
        sampled_sents = sents.add(Variable(corrupts)).remainder_(vocab_size)

        # converting sampled_sents into Variable before returning
        try:
            sampled_sents = Variable(sampled_sents)
        except:
            pass

        return sampled_sents

    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MBART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
          if self.da_method is not None:
            if self.da_method == 'replace':
              input_ids = self.word_replacement(input_ids)
              decoder_input_ids = self.word_replacement(labels)
            if self.da_method == 'switchout':
              input_ids = self.switch_out(input_ids, 0.3, self.config.vocab_size, self.bos_id, self.eos_id, self.pad_id)
              decoder_input_ids = self.switch_out(labels, 0.3, self.config.vocab_size, self.bos_id, self.eos_id, self.pad_id)
          else:
            input_ids = input_ids
            decoder_input_ids = labels

            decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
