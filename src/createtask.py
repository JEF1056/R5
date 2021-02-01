import io
import os
import json
import numpy as np
import sentencepiece as spm

with open("config.json", "r") as f:
    nq_tsv_path, max_len, out_dir=json.load(f)
sp = spm.SentencePieceProcessor(model_file=f"{os.path.join(out_dir,'bpe')}.model")

def stream(num_devices, split):
    with io.open(nq_tsv_path[split], mode="r", encoding="utf-8") as f:
        while True:
            inputs, in_mask, targets, tar_mask=[],[],[],[]
            for _ in range(num_devices):
                d=f.readline()
                if d == "": f.seek(0); d=f.readline()
                inp, tar= d.split("\t")
                inp, tar= sp.encode(inp), sp.encode(tar)
                inputs.append(np.asarray(np.pad(inp, (0, max_len-len(inp))), dtype=np.int32))
                in_mask.append(np.asarray(np.pad(np.ones_like(inp), (0, max_len-len(inp))), dtype=np.int32))
                targets.append(np.asarray(np.pad(tar, (0, max_len-len(tar))), dtype=np.int32))
                tar_mask.append(np.asarray(np.pad(np.ones_like(tar), (0, max_len-len(tar))), dtype=np.int32))
            inputs = np.stack(inputs)
            targets = np.stack(targets)
            in_mask = np.stack(in_mask)
            tar_mask = np.stack(tar_mask)
            yield inputs, targets, in_mask