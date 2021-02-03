import os
import json
import warnings
import argparse
import tensorflow as tf
import logging as py_logging
import src.helpers as helpers
from contextlib import contextmanager
from reformers.TFreformers import TFReformerLM, TFLSHAttention
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description='Train R5')
parser.add_argument('-dir', type=str, default="gs://conversation-R5",
                    help='link to google storage bucket')
parser.add_argument('-train', type=str, default="context-train.txt",
                    help='train file')
parser.add_argument('-val', type=str, default="context-val.txt",
                    help='val file')
parser.add_argument('-tpu_address', type=str, default=None,
                    help='TPU ip address')
parser.add_argument('-epochs', type=int, default=20,
                    help='# of epochs')
parser.add_argument('-validate_every', type=int, default=500,
                    help='# of epochs')
parser.add_argument('-max_len', type=int, default=2048,
                    help='max length for input/output sequences')
parser.add_argument('-lr', type=float, default=5e-6,
                    help='model learning rate')
parser.add_argument('-vocab_size', type=int, default=32768,
                    help='vocab size')
parser.add_argument('-steps', type=int, default=50000,
                    help='umber of steps to finetune')
parser.add_argument('-batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('-eval_steps', type=int, default=30,
                    help='number of samples to feed eval')
parser.add_argument("-eval", type=helpers.str2bool, nargs='?', const=True, default=False,
                    help="eval model after training")
args = parser.parse_args()

print("~~Parsing arguments~~")
with open("config.json", "w") as f:
    json.dump({"train":os.path.join(args.dir,"data", args.train), "validation": os.path.join(args.dir,"data", args.val)},f)
from src.createtask import nq_dataset_fn
if args.tpu_address != None: args.tpu_address = f"grpc://{args.tpu_address}:8470"

print("~~Setting up devices~~")
if args.tpu_address != None:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu_address)
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy=tf.distribute.TPUStrategy(tpu_cluster_resolver=tpu)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    #tf.disable_v2_behavior()
    
tf.get_logger().propagate = False
py_logging.root.setLevel('INFO')

@contextmanager
def tf_verbosity_level(level):
    og_level = tf.logging.get_verbosity()
    tf.logging.set_verbosity(level)
    yield
    tf.logging.set_verbosity(og_level)

print("~~Loading data~~")
train=nq_dataset_fn("train")
train=train.batch(args.batch_size)
train=train.prefetch(10)
val=nq_dataset_fn("validation")
val=val.batch(args.batch_size)
val=val.prefetch(10)

if args.tpu_address != None:
    with strategy.scope():
        #define data and model
        print("~~Setting up model~~")
        train=strategy.experimental_distribute_dataset(train)
        val=strategy.experimental_distribute_dataset(val)
        model_tf = TFReformerLM(
                num_tokens= args.vocab_size,
                emb = 512,
                depth = 6,   # batch 4 full attention 8 이면 안돌아감 
                max_seq_len = args.max_len,
                heads = 8,
                lsh_dropout = 0.1,
                causal = True,        # auto-regressive or not
                bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
                n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
                ff_chunks = 8,      # number of chunks for feedforward layer, make higher if there are memory issues
                weight_tie = True,   # tie parameters of each layer for no memory per additional depth
                attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
                use_full_attn = False   # use full self attention, for comparison
            )
        
        print(f"~~Begin Training for {args.epochs} epochs, {args.num_batches} steps per epoch~~")
        model_tf.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"], shuffle=True)
        model_tf.fit(train, batch_size=args.batch_size, epochs=args.epochs, validation_data=val)  
else:
    assert args.tpu_address != None, "non-TPU training is currently not implemented :3"