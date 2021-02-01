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
            inputs, targets=[],[]
            for _ in range(num_devices):
                d=f.readline()
                if d == "": f.seek(0); d=f.readline()
                inp, tar= d.split("\t")
                inputs.append(np.asarray(np.pad(sp.encode(inp), (0, max_len)), dtype=np.int32))
                targets.append(np.asarray(np.pad(sp.encode(tar), (0, max_len)), dtype=np.int32))
            inputs = np.stack(inputs)
            targets = np.stack(targets)
            yield inputs, targets