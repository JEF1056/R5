import os
import json
import functools
import numpy as np
import tensorflow as tf
import sentencepiece as spm
import tensorflow_datasets as tfds

with open("config.json", "r") as f:
    nq_tsv_path, max_len, out_dir=json.load(f)
sp = spm.SentencePieceProcessor(model_file=f"{os.path.join(out_dir,'bpe')}.model")
    
def preprocess(ds):
    def to_inputs_and_targets(ex):
        """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
        inp, tar= sp.encode(str(ex["question"])), sp.encode(str(ex["answer"]))
        if len(inp) < max_len or len(tar) < max_len:
            combined=inp+[1]+tar+[2]
            inp=np.asarray([np.pad(combined, (0, max_len-len(combined)))], dtype=np.int32)
        return tf.convert_to_tensor(inp,dtype=np.int32)
    return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def nq_dataset_fn(split, shuffle_files=False):
    # We only have one file for each split.
    del shuffle_files
    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(nq_tsv_path[split])
    # Split each "<question>\t<answer>" example into (question, answer) tuple.
    ds = ds.map(
    functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                      field_delim="\t", use_quote_delim=False),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex)))
    return preprocess(ds)

print("A few raw validation examples...")
for ex in tfds.as_numpy(nq_dataset_fn("validation").take(5)):
    print(ex)