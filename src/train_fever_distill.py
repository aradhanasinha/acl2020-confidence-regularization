"""Script to train BERT on Fever

Modified from the old "run_classifier" script from
https://github.com/huggingface/pytorch-transformer

Generally I tried not to change much, but I did add variable length sequence
encoding
and parallel pre-processing for the sake of performance

Fair warning, I probably broke some of the multi-gpu stuff, I have only tested
the single GPU version
"""

import argparse
import json
import logging
import os
import random
from collections import namedtuple
from os.path import join, exists
from typing import List, Dict, Iterable

# temporary hack for the pythonroot issue
import sys

import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, Sampler
from tqdm import trange, tqdm

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import config
import utils

import clf_distill_loss_functions
from bert_distill import BertDistill
from clf_distill_loss_functions import *

from train_utils import *

from predictions_analysis import visualize_predictions
from utils import Processor, process_par

LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
REV_LABEL_MAP = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

def load_fever(data_dir=config.FEVER_SOURCE, split="train", sample=None):
  if split == "train":
    filename = os.path.join(data_dir, "fever.train.jsonl")
  elif split == "dev":
    filename = os.path.join(data_dir, "fever.dev.jsonl")
  elif split == "symmetric_dev":
    filename = os.path.join(data_dir, "fever_symmetric_dev.jsonl")
  elif split == "symmetric_test":
    filename = os.path.join(data_dir, "fever_symmetric_test.jsonl")
  elif split == "generated":
    filename = os.path.join(data_dir, "fever_symmetric_generated.jsonl")
  elif split == "full":
    filename = os.path.join(data_dir, "fever_symmetric_full.jsonl")
  else:
    raise Exception("invalid split name")

  out = []
  logging.info("Loading jsonl from {}...".format(filename))
  with open(filename, "r") as jsonl_file:
    for i, line in enumerate(jsonl_file):
      example = json.loads(line)

      if "unique_id" in example:
        id = example["unique_id"]
      else:
        id = example["id"]

      hypothesis = example["claim"]
      try:
        evidence = example["evidence"]
        label = example["gold_label"]
      except:
        evidence = example["evidence_sentence"]
        label = example["label"]

      out.append(TextPairExample(id, hypothesis, evidence, LABEL_MAP[label]))

  if sample:
    random.shuffle(out)
    out = out[:sample]

  return out


def load_teacher_probs():
  file_path = config.FEVER_TEACHER_SOURCE

  with open(file_path, "r") as teacher_file:
    all_lines = teacher_file.read()
    all_json = json.loads(all_lines)

  return all_json


def load_bias(bias_name) -> Dict[str, np.ndarray]:
  """Load dictionary of example_id->bias where bias is a length 2 array

    of log-probabilities
  """
  if "fever_claim_only" in bias_name:
    file_path = config.BIAS_SOURCES[bias_name]

    with open(file_path, "r") as hypo_file:
      all_lines = hypo_file.read()
      bias = json.loads(all_lines)
      for k, v in bias.items():
        bias[k] = np.array(v)
    return bias
  else:
    raise Exception("invalid bias name")

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
      "--uniform_labeling_wt", default=0, type=float,
      help="The weight given to the uniform labeling regularization, currently used with shuffled examples.")
  parser.add_argument(
      "--do_eval",
      action="store_true",
      help="Whether to run eval on the dev set.")
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
          "none", "distill", "smoothed_distill", "theta_smoothed_distill",
          "reweight_baseline", "bias_product_baseline", "learned_mixin_baseline"
      ],
      default="learned_mixin",
      help="Kind of debiasing method to use")
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
      "--which_bias",
      choices=[
          "fever_claim_only", "fever_claim_only_bow",
          "fever_claim_only_infersent", "fever_claim_only_bow_reproduce",
          "fever_claim_only_balanced"
      ],
      default=None)
  parser.add_argument(
      "--theta",
      type=float,
      default=0.1,
      help="for theta smoothed distillation loss")

  args = parser.parse_args()

  utils.add_stdout_logger()

  if args.mode == "none":
    loss_fn = clf_distill_loss_functions.Plain()
  elif args.mode == "distill":
    loss_fn = clf_distill_loss_functions.DistillLoss()
  elif args.mode == "smoothed_distill":
    loss_fn = clf_distill_loss_functions.SmoothedDistillLoss()
  elif args.mode == "theta_smoothed_distill":
    loss_fn = clf_distill_loss_functions.ThetaSmoothedDistillLoss(args.theta)
  elif args.mode == "reweight_baseline":
    loss_fn = clf_distill_loss_functions.ReweightBaseline()
  elif args.mode == "bias_product_baseline":
    loss_fn = clf_distill_loss_functions.BiasProductBaseline()
  elif args.mode == "learned_mixin_baseline":
    loss_fn = clf_distill_loss_functions.LearnedMixinBaseline(args.penalty)
  else:
    raise RuntimeError("invalid mode")

  output_dir = args.output_dir

  if args.do_train:
    if exists(output_dir):
      if len(os.listdir(output_dir)) > 0:
        logging.warning("Output dir exists and is non-empty")
    else:
      os.makedirs(output_dir)

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

  if not args.do_train and not args.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  if os.path.exists(output_dir) and os.listdir(output_dir) and args.do_train:
    logging.warning(
        "Output directory ({}) already exists and is not empty.".format(
            output_dir))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
    train_examples = load_fever(
        split="train", sample=2000 if args.debug else None)
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
    train_features: List[InputFeatures] = convert_examples_to_features(
        train_examples, args.max_seq_length, tokenizer, args.n_processes)

    bias_map = None
    if args.mode != "none":
      if args.which_bias is None:
        raise Exception("bias source must be specified")

      bias_map = load_bias(args.which_bias)

      logging.info("**** filtering down the training example ****")
      logging.info("original len: {}".format(str(len(train_features))))
      train_features = [
          x for x in train_features if str(x.example_id) in bias_map
      ]
      logging.info("filtered len: {}".format(str(len(train_features))))

      for fe in train_features:
        fe.bias = bias_map[str(fe.example_id)].astype(np.float32)
      teacher_probs_map = load_teacher_probs()
      for fe in train_features:
        fe.teacher_probs = np.array(teacher_probs_map[str(
            fe.example_id)]).astype(np.float32)

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_examples))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)

    train_dataloader = build_train_dataloader(train_features,
                                              args.train_batch_size, args.seed,
                                              args.sorted)

    model.train()
    loss_ema = 0
    total_steps = 0
    decay = 0.99

    for _ in trange(int(args.num_train_epochs), desc="Epoch", ncols=100):
      tr_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0
      pbar = tqdm(train_dataloader, desc="loss", ncols=100)
      for step, batch in enumerate(pbar):
        new_batch = []
        for t in batch:
          if not isinstance(t, dict):
            new_batch.append(t.to(device))
          else:
            t_new = {k: [v.to(device) for v in t[k]] for k in t.keys()}
            new_batch.append(t_new)
        batch = tuple(new_batch)

        if bias_map is not None:
          example_ids, input_features_dict, bias, teacher_probs = batch

        else:
          bias = None
          teacher_probs = None
          example_ids, input_features_dict = batch
        input_ids, input_mask, segment_ids, label_ids = input_features_dict[
            InputFeatures.ORIGINAL_INPUT]

        logits, loss = model(input_ids, segment_ids, input_mask, label_ids,
                             bias, teacher_probs)
        #logging.warning(f"[ANUU DEBUG] loss {loss}, n_gpu {n_gpu}")

        if args.uniform_labeling_wt > 0:
          shuffled_input_ids, shuffled_input_mask, shuffled_segment_ids, _ = input_features_dict[
              InputFeatures.SHUFFLED_INPUT]
          shuffled_logits = model(shuffled_input_ids, shuffled_segment_ids,
                                  shuffled_input_mask, None, bias,
                                  teacher_probs)
          shuffled_loss_module = UniformLabelCrossEntropy()
          uniform_labeling_loss = shuffled_loss_module(
              shuffled_logits) * args.uniform_labeling_wt
          loss = torch.add(loss, uniform_labeling_loss)

        total_steps += 1
        loss_ema = loss_ema * decay + loss.cpu().detach().numpy() * (1 - decay)
        if n_gpu > 1:
          loss_ema_d = sum(loss_ema) / float(len(loss_ema))
        else:
          loss_ema_d = loss_ema
        descript = "loss=%.4f" % (loss_ema_d / (1 - decay**total_steps))
        pbar.set_description(descript, refresh=False)

        if n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu.
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
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    model = BertDistill(config, num_labels=3, loss_fn=loss_fn)
    model.load_state_dict(torch.load(output_model_file))

  model.to(device)

  if not args.do_eval:
    return
  if not (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    return

  model.eval()

  if args.do_eval:
    eval_datasets = [("fever_dev", load_fever(split="dev"))]
    eval_datasets += [("symmetric_test_v2", load_fever(split="symmetric_test"))]
    eval_datasets += [("symmetric_test_v1", load_fever(split="generated"))]

    # eval_datasets = [("symmetric_test_v2_dev", load_fever(split="symmetric_test"))]
    # eval_datasets += [("fever_train", load_fever(split="train"))]

  for ix, (name, eval_examples) in enumerate(eval_datasets):
    logging.info("***** Running evaluation on %s *****" % name)
    logging.info("  Num examples = %d", len(eval_examples))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_features = convert_examples_to_features(eval_examples,
                                                 args.max_seq_length, tokenizer)
    eval_features.sort()
    all_label_ids = np.array([x.get_original_label_id() for x in eval_features])
    eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

    eval_loss = 0
    nb_eval_steps = 0
    probs = []
    test_subm_ids = []

    for example_ids, input_ids, input_mask, segment_ids, label_ids in tqdm(
        eval_dataloader, desc="Evaluating", ncols=100):
      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)
      segment_ids = segment_ids.to(device)
      label_ids = label_ids.to(device)

      with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask)

      # create eval loss and other metric required by the task
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

    preds = np.argmax(probs, axis=1)

    result = {"accuracy": simple_accuracy(preds, all_label_ids)}
    # import pdb; pdb.set_trace()

    output_eval_file = os.path.join(output_dir, "eval_%s_results.txt" % name)
    with open(output_eval_file, "w") as writer:
      logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

    output_answer_file = os.path.join(output_dir, "eval_%s_answers.json" % name)
    answers = {
        ex.example_id: [float(x) for x in p
                       ] for ex, p in zip(eval_features, probs)
    }
    with open(output_answer_file, "w") as f:
      json.dump(answers, f)


if __name__ == "__main__":
  main()
