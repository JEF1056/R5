import gin
import trax
from trax.supervised import training
from trax import layers as tl
import argparse
import os
import json

parser = argparse.ArgumentParser(description='Export checkpoints for serving')
parser.add_argument('-dir', type=str, default="train",
                    help='Directory to save model')
parser.add_argument('-val', type=str, default="data/context-val.txt",
                    help='location of validation text')
parser.add_argument('-train', type=str, default="data/context-train.txt",
                    help='location of train text')
parser.add_argument('-max_length', type=int, default=4096,
                    help='maximum length for the model (ensure it matches gin)')
parser.add_argument('-vocab_size', type=int, default=32768,
                    help='bocab size for the model (ensure it matches gin)')
parser.add_argument('-tpu', type=str, default=None,
                    help='TPU ip address')
args = parser.parse_args()

gin.parse_config_file("config.gin")
if not os.path.exists(f"{os.path.join(args.dir,'bpe')}.model"):
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(input=args.train, model_prefix=os.path.join(args.dir,'bpe'), train_extremely_large_corpus=True, input_sentence_size=100000, shuffle_input_sentence=True, vocab_size=args.vocab_size, model_type="bpe", character_coverage = 1, user_defined_symbols=['\\n', "\\b"], bos_id=-1,eos_id=-1, pad_id=-1)

with open("config.json", "w") as f:
    json.dump([{"train":args.train, "validation": args.val}, args.max_length, args.dir], f)
from src.createtask import stream

if args.tpu != None:
    print("~~Setting Up Devices~~")
    import jax
    from jax.config import config
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = f"grpc://{args.tpu}:8470"
    print(config.FLAGS.jax_backend_target)
    print(f'{jax.host_count()} available devices')
    print(f'{jax.devices()} available cores')

print("~~Begin Training~~")
# Training task.
train_task = training.TrainTask(
    labeled_data=stream(trax.fastmath.device_count(), "train"),
    loss_layer=tl.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adafactor(),
    n_steps_per_checkpoint=2500,
)

# Evaluaton task.
eval_task = training.EvalTask(
    labeled_data=stream(trax.fastmath.device_count(), "validation"),
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    n_eval_batches=20  # For less variance in eval numbers.
)

output_dir = os.path.expanduser(args.dir)

# Train tiny model with Loop.
training_loop = training.Loop(
    trax.models.Reformer2(mode="train"),
    train_task,
    eval_tasks=[eval_task],
    output_dir=output_dir)

# run 1000 steps (batches)
training_loop.run(50000)