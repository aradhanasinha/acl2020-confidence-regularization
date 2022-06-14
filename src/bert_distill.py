from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch import nn, add, mul
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from clf_distill_loss_functions import ClfDistillLossFunction


class BertDistill(BertPreTrainedModel):
  """Pre-trained BERT model that uses our loss functions."""

  def __init__(self, config, num_labels, loss_fn: ClfDistillLossFunction):
    super(BertDistill, self).__init__(config)
    self.num_labels = num_labels
    self.loss_fn = loss_fn
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, num_labels)
    self.apply(self.init_bert_weights)

  def forward(self,
              input_ids,
              token_type_ids=None,
              attention_mask=None,
              labels=None,
              bias=None,
              teacher_probs=None):
    _, pooled_output = self.bert(
        input_ids,
        token_type_ids,
        attention_mask,
        output_all_encoded_layers=False)
    logits = self.classifier(self.dropout(pooled_output))
    if labels is None:
      return logits
    loss = self.loss_fn.forward(pooled_output, logits, bias, teacher_probs,
                                labels)
    return logits, loss

def get_embeddings(self,
                   input_ids,
                   token_type_ids=None,
                   attention_mask=None,
                   embedding_type="second_last_hidden"):
  outputs = self.model(
      inpute_ids=input_ids,
      token_type_ids=token_type_ids,
      attention_mask=attention_mask,
      output_hidden_states=True,
      return_dict=True)
  hidden_states = outputs.hidden_states

  if embedding_type == "last_hidden":
    return hidden_states[-1]

  if embedding_type == "second_last_hidden":
    return hidden_states[-2]

  if embedding_type == "avg_first_last":
    return mul(add(hidden_states[1], hidden_states[-1]), 0.5)

  if embedding_type == "avg_top2":
    return mul(add(hidden_states[-1], hidden_states[-2]), 0.5)

  raise NotImplementedError



