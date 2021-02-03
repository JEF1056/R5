import io
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
        print(ex["question"].numpy())
        inp, tar= sp.encode(ex["question"]), sp.encode(ex["answer"])
        if len(inp)+len(tar)+2 > max_len: inp=inp[max_len-len(inp)-len(tar)-2:]
        combined=inp+[1]+tar+[2]
        input=np.asarray(np.pad(combined, (0, max_len-len(combined))), dtype=np.int32)
        mask=np.asarray(np.pad(np.ones(len(tar)+1), (len(inp)+1, max_len-len(combined))), dtype=np.int32)
        return input,input,mask
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
    ds = ds.map(lambda *ex: dict(zip(["question", "answer"], tf.py_function(ex) )))
    return preprocess(ds)

print("A few raw validation examples...")
for ex in tfds.as_numpy(nq_dataset_fn("validation").take(5)):
    print(ex)
    print(f"{len(ex[0])}\n{len(ex[2])}")
    ovloc=sorted(np.where(ex[2]==1))
    print(ovloc)
    st=ovloc[0][0]-10 if ovloc[0][0]-10 <=0 else 0
    print(sp.decode(ex[0].tolist()[st:ovloc[0][0]+10]))
    overlay = [int(ex[0][i]) for i in ovloc[0]]
    print(f"Overlay: {overlay}\nDecoded: {sp.decode(overlay)}")