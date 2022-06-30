"""Tests for contrastive_loss."""

#from contrastive_loss import contrastive_loss
import torch
from contrastive_loss import transpose_last_two_dim, normalize_last_dim
import torch.nn.functional as F
import math 

def contrastive_loss(label_ids,
                     anchor,
                     positive_key,
                     negative_keys_paired=None,
                     use_unpaired_negative_keys=True,
                     temperature=0.1,
                     reduction='mean'):
  if negative_keys_paired is None and not use_unpaired_negative_keys:
    raise ValueError('Need negative examples: paired, unpaired or both.')
  if use_unpaired_negative_keys and not label_ids:
    raise ValueError('Need label_ids to implement unpaired negative keys')

  # Normalize D length embeddigngs to unit vectors, so we can dot product away.
  anchor, positive_key, negative_keys_paired = normalize_last_dim(
      anchor, positive_key, negative_keys_paired)

  if negative_keys_paired is not None:
    # Cosine between all anchor-negative combinations
    # (N, 1, M) <-- (N, 1, D) * (N, D, M)
    negative_logits = anchor.unsqueeze(1) @ transpose_last_two_dim(
        negative_keys_paired)
    negative_logits = negative_logits.squeeze(1)  # (N, M)

  if not use_unpaired_negative_keys:
    # Cosine between positive pairs
    # (N, 1) <-- (N, D) dot.prod (N, D), but keep dimension
    positive_logit = torch.sum(anchor * positive_key, dim=1, keepdim=True)

    # First index in last dimension are the positive samples
    # (N, M+1) <-- [positive, negative]
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    # Correct label is at the 0th index always
    labels = torch.zeros(len(logits), dtype=torch.long, device=anchor.device)
    # labels = torch.cat([
    #     torch.ones(positive_logit.size()),
    #     torch.zeros(negative_logits.size())
    # ], dim=1)
  else:
    # Cosine between all combinations
    # (N, N) <-- (N, D) * (D, N) (positive on diagonal)
    logits = anchor @ transpose_last_two_dim(positive_key)

    # Positive keys are the entries on the diagonal
    # And also the keys that share the same label.
    labels = torch.zeros(len(label_ids), len(label_ids))
    for label in label_ids.unique():
      examples_with_label = (label_ids == label).int()
      labels += examples_with_label.unsqueeze(1) * examples_with_label

    if negative_keys_paired is not None:
      # (N, N + M)  <-- [logits, negative]
      logits = torch.cat([logits, negative_logits], dim=1)
      labels = torch.cat([labels, torch.zeros(negative_logits.size())], dim=1)

  print('Logits', logits / temperature)
  print('Labels', labels)
  print('call cross entropy')
  ce = F.cross_entropy(logits / temperature, labels, reduction=reduction)
  return math.log(math.exp(ce) - 1)
  # Cross entropy includes the denominator and numerator of the contrastive Loss
  # in it's denominator, so this transform is to remove items from the same 
  # class from the denominator.


class ContrastiveLossTest():
  """
   Basic Unit Tests for personal sanity.
  """

  @staticmethod
  def isclose(a, b, tol=0.0001):
    # Only using four decimal points in my expected numbers below.
    return abs(a-b) <= tol

  def test_paired_contrastive_loss_single_example(self):
    label_ids = None
    use_unpaired_negative_keys = False
    temperature = 1

    anchor = torch.Tensor([[3., 4.]])
    positive_key = torch.Tensor([[6., 8.]])
    negative_keys = torch.Tensor([[[3., 4.], [0. , 0.], [5., 5.]]])

    result = contrastive_loss(label_ids, anchor, positive_key, negative_keys,
                              use_unpaired_negative_keys, temperature)


    expected_positive_logit = 1.0 # Positive logit is a match.
    numerator = math.exp(expected_positive_logit)

    expected_negative_logits = [1.0, 0.0, 0.9899]
    # Match, Null tensor returns 0, and CosineSimilarity[{5,5}, {3,4}] ==> 0.9899.
    denominator = sum([math.exp(l) for l in expected_negative_logits])
    expected_result = math.log(numerator / denominator) * -1
    # TODO(aradhanas): Fix mistake. numerator should not be in denominator.

    assert ContrastiveLossTest.isclose(expected_result, result)

  def run(self):
    self.test_paired_contrastive_loss_single_example()


if __name__ == '__main__':
  test = ContrastiveLossTest()
  test.run()
