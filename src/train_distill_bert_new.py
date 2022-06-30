"""Code for reproducing the results on our work in reducing bias via confidence regularization

The code is based on previous implementation by Clark et al. (2019)
https://github.com/chrisc36/debias/tree/master/debias

"""
import torch
from torch import nn
import pandas as pd

import argparse
import json
import logging
import os
import random
from os.path import join, exists
from typing import List, Dict, Iterable
import errno
from visualize_embeddings import reduce_embedding_dimensions, visualize, visualize_augmentations

# temporary hack for the pythonroot issue
import sys

import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss
from tqdm import trange, tqdm

import config
import utils

import clf_distill_loss_functions
from bert_distill import BertDistill
from clf_distill_loss_functions import *

from train_utils import *
from contrastive_loss import ContrastiveLoss

from predictions_analysis import visualize_predictions
from utils import Processor, process_par

HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"

NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(NLI_LABELS)}
NLI_LABEL_MAP["hidden"] = NLI_LABEL_MAP["entailment"]


def load_easy_hard():
  all_datasets = []

  all_datasets.append(("mnli_dev_matched_easy",
                       load_mnli(False, custom_path="dev_matched_easy.tsv")))
  all_datasets.append(("mnli_dev_matched_hard",
                       load_mnli(False, custom_path="dev_matched_hard.tsv")))
  all_datasets.append(("mnli_dev_mismatched_easy",
                       load_mnli(False, custom_path="dev_mismatched_easy.tsv")))
  all_datasets.append(("mnli_dev_mismatched_hard",
                       load_mnli(False, custom_path="dev_mismatched_hard.tsv")))

  return all_datasets


def load_hans_subsets():
  src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
  if not exists(src):
    logging.info("Downloading source to %s..." % config.HANS_SOURCE)
    utils.download_to_file(HANS_URL, src)

  hans_datasets = []
  labels = ["entailment", "non-entailment"]
  subsets = set()
  with open(src, "r") as f:
    for line in f.readlines()[1:]:
      line = line.split("\t")
      subsets.add(line[-3])
  subsets = [x for x in subsets]

  for label in labels:
    for subset in subsets:
      name = "hans_{}_{}".format(label, subset)
      examples = load_hans(filter_label=label, filter_subset=subset)
      hans_datasets.append((name, examples))

  return hans_datasets


def load_hans(n_samples=None,
              filter_label=None,
              filter_subset=None) -> List[TextPairExample]:
  out = []

  if filter_label is not None and filter_subset is not None:
    logging.info("Loading hans subset: {}-{}...".format(filter_label,
                                                        filter_subset))
  else:
    logging.info("Loading hans all...")

  src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
  if not exists(src):
    logging.info("Downloading source to %s..." % config.HANS_SOURCE)
    utils.download_to_file(HANS_URL, src)

  with open(src, "r") as f:
    f.readline()
    lines = f.readlines()

  if n_samples is not None:
    lines = np.random.RandomState(16349 + n_samples).choice(
        lines, n_samples, replace=False)

  for line in lines:
    parts = line.split("\t")
    label = parts[0]

    if filter_label is not None and filter_subset is not None:
      if label != filter_label or parts[-3] != filter_subset:
        continue

    if label == "non-entailment":
      label = 0
    elif label == "entailment":
      label = 1
    else:
      raise RuntimeError()
    s1, s2, pair_id = parts[5:8]
    out.append(TextPairExample(pair_id, s1, s2, label))
  return out


def ensure_mnli_is_downloaded():
  mnli_source = config.GLUE_SOURCE
  logging.warning(
      f"[ANUU DEBUG] ensure_mnli_is_downloaded.mnli_source {mnli_source}")
  logging.warning(
      f"[ANUU DEBUG] ensure_mnli_is_downloaded.mnli_source exists {exists(mnli_source)}"
  )
  logging.warning(
      f"[ANUU DEBUG] ensure_mnli_is_downloaded.mnli_source os.listdir {os.listdir(mnli_source)}"
  )

  if exists(mnli_source) and len(os.listdir(mnli_source)) > 0:
    return
  else:
    raise Exception("Download MNLI from Glue and put files under glue_multinli")


def load_mnli(is_train, sample=None, custom_path=None) -> List[TextPairExample]:
  ensure_mnli_is_downloaded()
  if is_train:
    filename = join(config.GLUE_SOURCE, "train.tsv")
  else:
    if custom_path is None:
      filename = join(config.GLUE_SOURCE, "dev_matched.tsv")
    else:
      filename = join(config.GLUE_SOURCE, custom_path)

  logging.info("Loading mnli " + ("train" if is_train else "dev"))
  with open(filename) as f:
    f.readline()
    lines = f.readlines()

  if sample:
    lines = np.random.RandomState(26096781 + sample).choice(
        lines, sample, replace=False)

  out = []
  for line in lines:
    line = line.split("\t")
    out.append(
        TextPairExample(line[0], line[8], line[9],
                        NLI_LABEL_MAP[line[-1].rstrip()]))
  return out


def load_teacher_probs(custom_teacher=None):
  if custom_teacher is None:
    file_path = config.TEACHER_SOURCE
  else:
    file_path = custom_teacher

  with open(file_path, "r") as teacher_file:
    all_lines = teacher_file.read()
    all_json = json.loads(all_lines)

  return all_json


def load_bias(bias_name) -> Dict[str, np.ndarray]:
  """Load dictionary of example_id->bias where bias is a length 3 array

    of log-probabilities
  """

  if bias_name == "hans":
    if bias_name == "hans":
      bias_src = config.MNLI_WORD_OVERLAP_BIAS
    if not exists(bias_src):
      raise Exception("lexical overlap bias file is not found")
    bias = utils.load_pickle(bias_src)
    for k, v in bias.items():
      # Convert from entail vs non-entail to 3-way classes by splitting non-entail
      # to neutral and contradict
      bias[k] = np.array([
          v[0] - np.log(2.),
          v[1],
          v[0] - np.log(2.),
      ])
    return bias

  if bias_name in config.BIAS_SOURCES:
    file_path = config.BIAS_SOURCES[bias_name]
    with open(file_path, "r") as hypo_file:
      all_lines = hypo_file.read()
      bias = json.loads(all_lines)
      for k, v in bias.items():
        bias[k] = np.array(v)
    return bias
  else:
    raise Exception("invalid bias name")


def load_all_test_jsonl():
  test_datasets = []
  test_datasets.append(("mnli_test_m",
                        load_jsonl("multinli_0.9_test_matched_unlabeled.jsonl",
                                   config.MNLI_TEST_SOURCE)))
  test_datasets.append(
      ("mnli_test_mm",
       load_jsonl("multinli_0.9_test_mismatched_unlabeled.jsonl",
                  config.MNLI_TEST_SOURCE)))
  test_datasets.append(
      ("mnli_test_hard_m",
       load_jsonl("multinli_0.9_test_matched_unlabeled_hard.jsonl",
                  config.MNLI_HARD_SOURCE)))
  test_datasets.append(
      ("mnli_test_hard_mm",
       load_jsonl("multinli_0.9_test_mismatched_unlabeled_hard.jsonl",
                  config.MNLI_HARD_SOURCE)))
  return test_datasets


def load_jsonl(file_path, data_dir, sample=None):
  out = []
  full_path = join(data_dir, file_path)
  logging.info("Loading jsonl from {}...".format(full_path))
  with open(full_path, "r") as jsonl_file:
    for _, line in enumerate(jsonl_file):
      example = json.loads(line)

      label = example["gold_label"]
      if label == "-":
        continue

      line_id = example["pairID"]
      text_a = example["sentence1"]
      text_b = example["sentence2"]

      out.append(TextPairExample(line_id, text_a, text_b, NLI_LABEL_MAP[label]))

  if sample:
    random.shuffle(out)
    out = out[:sample]

  return out


def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument(
      "--bert_model",
      default="bert-base-uncased",
      type=str,
      help="Bert pre-trained model selected in the list: bert-base-uncased, "
      "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
      "bert-base-multilingual-cased, bert-base-chinese.")
  parser.add_argument(
      "--output_dir",
      default=None,
      type=str,
      required=True,
      help="The output directory where the model predictions and checkpoints will be written."
  )
  parser.add_argument(
      "--cache_dir",
      default="",
      type=str,
      help="Where do you want to store the pre-trained models downloaded from s3"
  )
  parser.add_argument(
      "--max_seq_length",
      default=128,
      type=int,
      help="The maximum total input sequence length after WordPiece tokenization. \n"
      "Sequences longer than this will be truncated, and sequences shorter \n"
      "than this will be padded.")
  parser.add_argument(
      "--do_train", action="store_true", help="Whether to run training.")
  parser.add_argument(
      "--uniform_labeling_wt",
      default=0,
      type=float,
      help="The weight given to the uniform labeling regularization, currently used with shuffled examples."
  )
  parser.add_argument(
      "--contrastive_loss_wt",
      default=0,
      type=float,
      help="The weight given to the contrastive loss fn, currently used with shuffled examples as negatives and dropouts as positives."
  )
  parser.add_argument(
      "--contrastive_loss_temp",
      default=0,
      type=float,
      help="The temperature for the contrastive loss fn.")
  parser.add_argument(
      "--use_unpaired_negative_keys",
      action="store_true",
      help="Whether the negative examples in the contrastive loss are also including the positive examples for other examples."
  )
  parser.add_argument(
      "--use_paired_negative_keys",
      action="store_true",
      help="Whether the negative examples in the contrastive loss are also including the positive examples for other examples."
  )
  parser.add_argument(
      "--do_eval",
      action="store_true",
      help="Whether to run eval on the dev set.")
  parser.add_argument(
      "--do_viz",
      action="store_true",
      help="Whether to visualize the embeddings.")
  parser.add_argument(
      "--do_test",
      action="store_true",
      help="Whether to run test and create submission.")
  parser.add_argument(
      "--train_batch_size",
      default=32,
      type=int,
      help="Total batch size for training.")
  parser.add_argument(
      "--seed",
      default=None,
      type=int,
      help="Seed for randomized elements in the training")
  parser.add_argument(
      "--eval_batch_size",
      default=16,
      type=int,
      help="Total batch size for eval.")
  parser.add_argument(
      "--learning_rate",
      default=5e-5,
      type=float,
      help="The initial learning rate for Adam.")
  parser.add_argument(
      "--num_train_epochs",
      default=3.0,
      type=float,
      help="Total number of training epochs to perform.")
  parser.add_argument(
      "--warmup_proportion",
      default=0.1,
      type=float,
      help="Proportion of training to perform linear learning rate warmup for. "
      "E.g., 0.1 = 10%% of training.")
  parser.add_argument(
      "--no_cuda",
      action="store_true",
      help="Whether not to use CUDA when available")
  parser.add_argument(
      "--local_rank",
      type=int,
      default=-1,
      help="local_rank for distributed training on gpus")
  parser.add_argument(
      "--gradient_accumulation_steps",
      type=int,
      default=1,
      help="Number of updates steps to accumulate before performing a backward/update pass."
  )
  parser.add_argument(
      "--fp16",
      action="store_true",
      help="Whether to use 16-bit float precision instead of 32-bit")
  parser.add_argument(
      "--loss_scale",
      type=float,
      default=0,
      help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
      "0 (default value): dynamic loss scaling.\n"
      "Positive power of 2: static loss scaling value.\n")

  ## Our arguements
  parser.add_argument(
      "--mode",
      choices=[
          "none", "distill", "smoothed_distill", "reweight_baseline",
          "bias_product_baseline", "learned_mixin_baseline"
      ])
  parser.add_argument(
      "--penalty",
      type=float,
      default=0.03,
      help="Penalty weight for the learn_mixin model")
  parser.add_argument(
      "--n_processes",
      type=int,
      default=4,
      help="Processes to use for pre-processing")
  parser.add_argument("--debug", action="store_true")
  parser.add_argument(
      "--sorted",
      action="store_true",
      help="Sort the data so most batches have the same input length,"
      " makes things about 2x faster. Our experiments did not actually"
      " use this in the end (not sure if it makes a difference) so "
      "its off by default.")
  parser.add_argument(
      "--which_bias", choices=["hans", "hypo", "hans_json"], required=True)
  parser.add_argument("--custom_teacher", default=None)
  parser.add_argument(
      "--theta",
      type=float,
      default=0.1,
      help="for theta smoothed distillation loss")

  args = parser.parse_args()

  utils.add_stdout_logger()
  logging.warning(f"[ANUU DEBUG] args: {args}")

  if args.mode == "none":
    loss_fn = clf_distill_loss_functions.Plain()
  elif args.mode == "distill":
    loss_fn = clf_distill_loss_functions.DistillLoss()
  elif args.mode == "smoothed_distill":
    loss_fn = clf_distill_loss_functions.SmoothedDistillLoss()
  elif args.mode == "reweight_baseline":
    loss_fn = clf_distill_loss_functions.ReweightBaseline()
  elif args.mode == "bias_product_baseline":
    loss_fn = clf_distill_loss_functions.BiasProductBaseline()
  elif args.mode == "learned_mixin_baseline":
    loss_fn = clf_distill_loss_functions.LearnedMixinBaseline(args.penalty)
  else:
    raise RuntimeError()

  output_dir = args.output_dir

  if args.do_train:
    if exists(output_dir):
      if len(os.listdir(output_dir)) > 0:
        logging.warning("Output dir exists and is non-empty")
    else:
      try:
        os.makedirs(output_dir)
      except OSError as e:
        logging.warning(f"[ANUU DEBUG] OSError: {e}")
        if e.errno == errno.EEXIST:
          print("Directory not created.")
          logging.warning(f"[ANUU DEBUG] OSError: {e}")
        else:
          raise

  print("Saving model to %s" % output_dir)

  if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
  else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
  logging.info(
      "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}"
      .format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

  if args.gradient_accumulation_steps < 1:
    raise ValueError(
        "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
        .format(args.gradient_accumulation_steps))

  args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

  if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
      torch.cuda.manual_seed_all(args.seed)

  if not args.do_train and not args.do_eval and not args.do_test:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  if os.path.exists(output_dir) and os.listdir(output_dir) and args.do_train:
    logging.warning(
        "Output directory ({}) already exists and is not empty.".format(
            output_dir))
  if not os.path.exists(output_dir):
    try:
      os.makedirs(output_dir)
    except OSError as e:
      logging.warning(f"[ANUU DEBUG] OSError: {e}")
      if e.errno == errno.EEXIST:
        print("Directory not created.")
        logging.warning(f"[ANUU DEBUG] OSError: {e}")
      else:
        raise

  # Its way ot easy to forget if this is being set by a command line flag
  if "-uncased" in args.bert_model:
    do_lower_case = True
  elif "-cased" in args.bert_model:
    do_lower_case = False
  else:
    raise NotImplementedError(args.bert_model)

  tokenizer = BertTokenizer.from_pretrained(
      args.bert_model, do_lower_case=do_lower_case)

  num_train_optimization_steps = None
  train_examples = None
  if args.do_train:
    logging.warning(f"[ANUU DEBUG] Calling load_mnli")
    train_examples = load_mnli(True, 2000 if args.debug else None)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size /
        args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
      num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size(
      )

  # Prepare model
  cache_dir = args.cache_dir if args.cache_dir else os.path.join(
      str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_{}".format(
          args.local_rank))

  model = BertDistill.from_pretrained(
      args.bert_model, cache_dir=cache_dir, num_labels=3, loss_fn=loss_fn)

  if args.fp16:
    model.half()
  model.to(device)
  if args.local_rank != -1:
    try:
      from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
      raise ImportError(
          "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
      )

    model = DDP(model)
  elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

  # Prepare optimizer
  param_optimizer = list(model.named_parameters())
  no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [{
      "params": [
          p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
      ],
      "weight_decay": 0.01
  }, {
      "params": [
          p for n, p in param_optimizer if any(nd in n for nd in no_decay)
      ],
      "weight_decay": 0.0
  }]
  if args.fp16:
    try:
      from apex.optimizers import FP16_Optimizer
      from apex.optimizers import FusedAdam
    except ImportError:
      raise ImportError(
          "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
      )

    optimizer = FusedAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        bias_correction=False,
        max_grad_norm=1.0)
    if args.loss_scale == 0:
      optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
      optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

  else:
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        t_total=num_train_optimization_steps)

  global_step = 0
  nb_tr_steps = 0
  tr_loss = 0

  if args.do_train:
    #TODO(aradhanas): Modify the function below to take in create_shuffled_examples and act on it.
    train_features: List[InputFeatures] = convert_examples_to_features(
        train_examples, args.max_seq_length, tokenizer, args.n_processes)

    if args.which_bias == "mix":
      hypo_bias_map = load_bias("hypo")
      hans_bias_map = load_bias("hans")
      bias_map = {}

      def compute_entropy(probs, base=3):
        return -(probs * (np.log(probs) / np.log(base))).sum()

      for key in hypo_bias_map.keys():
        hypo_ent = compute_entropy(np.exp(hypo_bias_map[key]))
        hans_ent = compute_entropy(np.exp(hans_bias_map[key]))
        if hypo_ent < hans_ent:
          bias_map[key] = hypo_bias_map[key]
        else:
          bias_map[key] = hans_bias_map[key]
    else:
      bias_map = load_bias(args.which_bias)

    for fe in train_features:
      fe.bias = bias_map[fe.example_id].astype(np.float32)
    teacher_probs_map = load_teacher_probs(args.custom_teacher)
    for fe in train_features:
      fe.teacher_probs = np.array(teacher_probs_map[fe.example_id]).astype(
          np.float32)

    example_map = {}
    for ex in train_examples:
      example_map[ex.id] = ex

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_examples))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)

    train_dataloader = build_train_dataloader(train_features,
                                              args.train_batch_size, args.seed,
                                              args.sorted)

    model.train()
    loss_ema = 0
    uniform_loss_ema = 0
    contrastive_loss_ema = 0
    total_steps = 0
    decay = 0.99

    training_losses = []
    for epoch_index in trange(int(args.num_train_epochs), desc="Epoch", ncols=100):
      tr_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0
      pbar = tqdm(train_dataloader, desc="loss", ncols=100)
      for step, batch in enumerate(pbar):
        new_batch = []
        for t in batch:
          if not isinstance(t, dict):
            new_batch.append(t.to(device))
          else:
            t_new = {}
            for k in t.keys():
              if k == InputFeatures.ORIGINAL_INPUT:
                val = [v.to(device) for v in t[k]]
              else:
                val = [[v.to(device) for v in x] for x in t[k]]
              t_new[k] = val
            new_batch.append(t_new)
        batch = tuple(new_batch)

        if bias_map is not None:
          example_ids, input_features_dict, bias, teacher_probs = batch

        else:
          bias = None
          example_ids, input_features_dict = batch
        input_ids, input_mask, segment_ids, label_ids = input_features_dict[
            InputFeatures.ORIGINAL_INPUT]

        logits, loss = model(input_ids, segment_ids, input_mask, label_ids,
                             bias, teacher_probs)
        #logging.warning(f"[ANUU DEBUG] loss {loss}, n_gpu {n_gpu}")

        if args.uniform_labeling_wt > 0:
          shuffled_input_ids, shuffled_input_mask, shuffled_segment_ids, _ = input_features_dict[
              InputFeatures.SHUFFLED_INPUT_LIST][0]
          shuffled_logits = model(shuffled_input_ids, shuffled_segment_ids,
                                  shuffled_input_mask, None, bias,
                                  teacher_probs)
          shuffled_loss_module = UniformLabelCrossEntropy()
          uniform_labeling_loss = shuffled_loss_module(
              shuffled_logits) * args.uniform_labeling_wt
        else:
          uniform_labeling_loss = 0

        #TODO(aradhanas): Add this arg and implement.
        if args.contrastive_loss_wt > 0:
          anchor_embedding = model(
              input_ids, segment_ids, input_mask, return_embedding=True)

          def get_embedding_list(input_list_name):
            embedding_list = []
            for val in input_features_dict[input_list_name]:
              input_ids, input_mask, segment_ids, _ = val
              embedding = model(
                  input_ids, segment_ids, input_mask, return_embedding=True)
              embedding_list.append(embedding)
            return embedding_list

          shuffled_embeddings = get_embedding_list(
              InputFeatures.SHUFFLED_INPUT_LIST)
          token_drop_embeddings = get_embedding_list(
              InputFeatures.TOKEN_DROPOUT_INPUTS_LIST)

          contrastive_loss_module = ContrastiveLoss(
              args.contrastive_loss_temp, args.use_unpaired_negative_keys)
          contrastive_loss = contrastive_loss_module(
              label_ids, anchor_embedding,
              shuffled_embeddings if args.use_paired_negative_keys else None,
              token_drop_embeddings) * args.contrastive_loss_wt
        else:
          contrastive_loss = 0

        total_steps += 1

        # Loss EMAs (not used for training.)
        def get_updated_loss_ema(new_loss, loss_ema):
          return loss_ema * decay + new_loss * (1 - decay)
        loss_ema = get_updated_loss_ema(loss.cpu().detach().numpy(), loss_ema)
        contrastive_loss_ema = get_updated_loss_ema(contrastive_loss,
                                                    contrastive_loss_ema)
        uniform_loss_ema = get_updated_loss_ema(uniform_labeling_loss,
                                                uniform_loss_ema)
        if n_gpu > 1:
          loss_ema_d = sum(loss_ema) / float(len(loss_ema))
        else:
          loss_ema_d = loss_ema
        # Get description with EMAs
        get_decay_loss = lambda loss_ema: loss_ema / (1 - decay**total_steps)
        descrip = f"EMA:loss={get_decay_loss(loss_ema_d):.{4}f}"
        if args.contrastive_loss_wt > 0:
          descrip = descrip + f",cl={get_decay_loss(contrastive_loss_ema):.{4}f}"
        if args.uniform_labeling_wt > 0:
          descrip = descrip + f",ul={get_decay_loss(uniform_loss_ema):.{4}f}"
        pbar.set_description(descrip, refresh=False)

        # Get losses used for training.
        if n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu.
        loss = torch.add(loss, contrastive_loss)
        loss = torch.add(loss, uniform_labeling_loss)
        training_losses.append(
            [epoch_index, step, loss, contrastive_loss, uniform_labeling_loss])

        if args.gradient_accumulation_steps > 1:
          loss = loss / args.gradient_accumulation_steps

        if args.fp16:
          optimizer.backward(loss)
        else:
          loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
          if args.fp16:
            # modify learning rate with special warm up BERT uses
            # if args.fp16 is False, BertAdam is used that handles this automatically
            lr_this_step = args.learning_rate * warmup_linear(
                global_step / num_train_optimization_steps,
                args.warmup_proportion)
            for param_group in optimizer.param_groups:
              param_group["lr"] = lr_this_step
          optimizer.step()
          optimizer.zero_grad()
          global_step += 1

    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(
        model, "module") else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    with open(output_config_file, "w") as f:
      f.write(model_to_save.config.to_json_string())
    output_train_loss_file = os.path.join(output_dir, "training_loss_over_time")
    with open(output_train_loss_file, "w") as f:
      header = "Epoch, Step, CE_loss, Contrastive_loss, Uniform_labeling_loss"
      f.write(header)
      for entry in training_losses:
        f.write(",".join([str(e) for e in entry])+ "\n")

    # Record the args as well
    arg_dict = {}
    for arg in vars(args):
      arg_dict[arg] = getattr(args, arg)
    with open(join(output_dir, "args.json"), "w") as out_fh:
      json.dump(arg_dict, out_fh)

    # Load a trained model and config that you have fine-tuned
    config = BertConfig(output_config_file)
    model = BertDistill(config, num_labels=3, loss_fn=loss_fn)
    model.load_state_dict(torch.load(output_model_file))
  else:
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    config = BertConfig.from_json_file(output_config_file)
    config = BertConfig.from_json_file(output_config_file)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    config = BertConfig.from_json_file(output_config_file)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    model = BertDistill(config, num_labels=3, loss_fn=loss_fn)
    model.load_state_dict(torch.load(output_model_file))

  model.to(device)

  if not args.do_eval and not args.do_test:
    return
  if not (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    return

  model.eval()

  if args.do_eval:
    eval_datasets = [("mnli_dev_m", load_mnli(False)),
                     ("mnli_dev_mm",
                      load_mnli(False, custom_path="dev_mismatched.tsv"))]
    eval_datasets += load_easy_hard()
    eval_datasets += [("hans", load_hans())]
    eval_datasets += load_hans_subsets()
  else:
    eval_datasets = []

  if args.do_test:
    test_datasets = load_all_test_jsonl()
    eval_datasets += test_datasets
    subm_paths = ["../submission/{}.csv".format(x[0]) for x in test_datasets]

  for ix, (name, eval_examples) in enumerate(eval_datasets):
    logging.info("***** Running evaluation on %s *****" % name)
    logging.info("  Num examples = %d", len(eval_examples))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_features = convert_examples_to_features(eval_examples,
                                                 args.max_seq_length, tokenizer)
    #TODO(aradhanas): Update to use the new InputFeatures.
    eval_features.sort()
    all_label_ids = np.array([x.get_original_label_id() for x in eval_features])
    eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

    eval_loss = 0
    nb_eval_steps = 0
    probs = []
    test_subm_ids = []
    
    if args.do_viz:
      # Lists to store reduced embeddigngs to visualize.
      original_reduced_embed_dfs = []
      dropout_reduced_embed_dfs = []
      shuffled_reduced_embed_dfs = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=100):
      example_ids, input_features_dict = batch
      input_ids, input_mask, segment_ids, label_ids = input_features_dict[
          InputFeatures.ORIGINAL_INPUT]

      if args.do_viz:
        dropout_input_ids, dropout_input_mask, dropout_segment_ids, _ = input_features_dict[
            InputFeatures.TOKEN_DROPOUT_INPUTS_LIST][0]
        shuffled_input_ids, shuffled_input_mask, shuffled_segment_ids, _ = input_features_dict[
            InputFeatures.SHUFFLED_INPUT_LIST][0]

      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)
      segment_ids = segment_ids.to(device)
      label_ids = label_ids.to(device)

      with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask)

        # Compute reduced dimension embeddigngs.
        if args.do_viz:
          embedding = model(
              input_ids, segment_ids, input_mask, return_embedding=True)
          original_reduced_embed_dfs.append(
              reduce_embedding_dimensions(embedding, input_mask, label_ids))
          dropout_embedding = model(
              dropout_input_ids,
              dropout_segment_ids,
              dropout_input_mask,
              return_embedding=True)
          dropout_reduced_embed_dfs.append(dropout_embedding,
                                           dropout_input_mask, label_ids)
          shuffled_embedding = model(
              shuffled_input_ids,
              shuffled_segment_ids,
              shuffled_input_mask,
              return_embedding=True)
          shuffled_reduced_embed_dfs.append(shuffled_embedding,
                                            shuffled_input_mask, label_ids)

      # Create eval loss and other metric required by the task.
      loss_fct = CrossEntropyLoss()
      tmp_eval_loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))

      eval_loss += tmp_eval_loss.mean().item()
      nb_eval_steps += 1
      probs.append(
          torch.nn.functional.softmax(logits, 1).detach().cpu().numpy())
      test_subm_ids.append(example_ids.cpu().numpy())

    probs = np.concatenate(probs, 0)
    test_subm_ids = np.concatenate(test_subm_ids, 0)
    eval_loss = eval_loss / nb_eval_steps

    if "hans" in name:
      # take max of non-entailment rather than taking their sum
      probs[:, 0] = probs[:, [0, 2]].max(axis=1)
      probs = probs[:, :2]

    preds = np.argmax(probs, axis=1)

    # Collate dfs and visualize.
    if args.do_viz:
      original_reduced_embed_df = pd.concat(
          original_reduced_embed_dfs, ignore_index=True)
      visualize(original_reduced_embed_df, output_dir, name)
      dropout_reduced_embed_df = pd.concat(
          dropout_reduced_embed_dfs, ignore_index=True)
      shuffled_reduced_embed_df = pd.concat(
          shuffled_reduced_embed_dfs, ignore_index=True)
      visualize_augmentations(original_reduced_embed_df,
                              dropout_reduced_embed_df,
                              shuffled_reduced_embed_df, output_dir, name)

    result = {"acc": simple_accuracy(preds, all_label_ids)}
    result["loss"] = eval_loss

    conf_plot_file = os.path.join(output_dir, "eval_%s_confidence.png" % name)
    ece, bins_acc, bins_conf, bins_num = visualize_predictions(
        probs, all_label_ids, conf_plot_file=conf_plot_file)
    result["ECE"] = ece
    result["bins_acc"] = bins_acc
    result["bins_conf"] = bins_conf
    result["bins_num"] = bins_num

    output_eval_file = os.path.join(output_dir, "eval_%s_results.txt" % name)
    output_all_eval_file = os.path.join(output_dir, "eval_all_results.txt")

    with open(output_eval_file, "w") as writer, open(output_all_eval_file,
                                                     "a") as all_writer:
      logging.info("***** Eval results *****")
      all_writer.write("eval results on %s:\n" % name)
      for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
        all_writer.write("%s = %s\n" % (key, str(result[key])))

    output_answer_file = os.path.join(output_dir, "eval_%s_answers.json" % name)
    answers = {
        ex.example_id: [float(x) for x in p
                       ] for ex, p in zip(eval_features, probs)
    }
    with open(output_answer_file, "w") as f:
      json.dump(answers, f)

    # prepare submission file
    if args.do_test and ix >= len(eval_datasets) - len(test_datasets):
      with open(subm_paths.pop(0), "w") as subm_f:
        subm_f.write("pairID,gold_label\n")
        for sub_id, pred_label_id in zip(test_subm_ids, preds):
          subm_f.write("{},{}\n".format(
              str(sub_id), REV_NLI_LABEL_MAP[pred_label_id]))


if __name__ == "__main__":
  main()
