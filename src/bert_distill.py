from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch import nn, add, mul

from clf_distill_loss_functions import ClfDistillLossFunction


class BertDistill(BertPreTrainedModel):
  """Pre-trained BERT model that uses our loss functions."""

  def __init__(self, config, num_labels, loss_fn: ClfDistillLossFunction):
    super(BertDistill, self).__init__(config)
    self.num_labels = num_labels
    self.loss_fn = loss_fn

    print(config)
    model = BertModel(config)
    self.bert = model
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, num_labels)
    self.apply(self.init_bert_weights)

  def forward(self,
              input_ids,
              token_type_ids=None,
              attention_mask=None,
              labels=None,
              bias=None,
              teacher_probs=None,
              return_embedding=False):

    if return_embedding:
        return self.get_embeddings(input_ids, token_type_ids, attention_mask)
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
                   embedding_type="last_hidden"):
    hidden_states, _ = self.bert(
      input_ids=input_ids,
      token_type_ids=token_type_ids,
      attention_mask=attention_mask)

    if embedding_type == "last_hidden":
      return hidden_states[-1]

    if embedding_type == "second_last_hidden":
      return hidden_states[-2]

    if embedding_type == "avg_first_last":
      return mul(add(hidden_states[1], hidden_states[-1]), 0.5)

    if embedding_type == "avg_top2":
      return mul(add(hidden_states[-1], hidden_states[-2]), 0.5)

    raise NotImplementedError



