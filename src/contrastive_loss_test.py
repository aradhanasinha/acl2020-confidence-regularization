"""Tests for contrastive_loss."""

from contrastive_loss import contrastive_loss
import torch
from contrastive_loss import ContrastiveLoss
import torch.nn.functional as F
import math 

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
    print(result)


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
