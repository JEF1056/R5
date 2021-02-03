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
parser.add_argument('-epochs', type=int, default=100000,
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
parser.add_argument('-batch_size', type=int, default=4,
                    help='batch size')
parser.add_argument('-num_batches', type=int, default=100000,
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
    tf.config.experimental_run_functions_eagerly(False)
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy=tf.distribute.TPUStrategy(tpu_cluster_resolver=tpu)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    tf.disable_v2_behavior()
    
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

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none', name='loss')
        accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        
        model_tf.set_optimizer(tf.keras.optimizers.Adam(args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8))
        model_tf.create_checkpoint_manager(os.path.join(args.dir,"models","r5"), max_to_keep=5, load_model=False)
        
        print(f"~~Begin Training for {args.epochs} epochs, {args.num_batches} steps per epoch~~")
        for e in range(1,args.epochs+1):
            for (step, (inputs, targets)) in enumerate(train):
                step = (e) * (step+1)
                loss = model_tf.train_step(inputs,targets,loss_object,train_loss,strategy,distributed=True)
                tf.print(step,loss)
                    
                if step % 1000 == 0:
                    ckpt_save_path = model_tf.ckpt_manager.save()
                    print('Saving checkpoint for step {} at {}'.format(step, ckpt_save_path))

                    if step % args.validate_every == 0:
                        total_eval_loss=0
                        for (eval_step, (inputs_val, targets_val)) in enumerate(val):
                            if eval_step==args.eval_steps: break
                            eval_loss = model_tf.eval_step(inputs_val,targets_val,loss_object,train_loss,strategy,distributed=True)
                            total_eval_loss+=eval_loss
                        print("eval loss",total_eval_loss/float(eval_step+1))

                    """
                    if step % GENERATE_EVERY == 0:
                        print("generate")
                        asdf = sampler_dataset_val[0][:-1]
                        print(asdf)
                        generated_seq = sg.sample_sequence(asdf,
                                                        predict_len=30,
                                                        temperature=1.0,
                                                        top_k=8,
                                                        top_p=0.9,
                                                        nucleus_sampling=True)

                        print("Generated seq by model:- " + generated_seq) 
                    """    

                if step>args.num_batches:
                    break        
else:
    assert args.tpu_address != None, "non-TPU training is currently not implemented :3"