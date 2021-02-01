import io
import os
import json
import numpy as np
import sentencepiece as spm

nq_tsv_path, max_len, out_dir=json.load(io.open("config.json", "r"))
sp = spm.SentencePieceProcessor(model_file=f"{os.path.join(out_dir,'bpe')}.model")

def stream(num_devices, split):
    with io.open(nq_tsv_path[split], mode="r", encoding="utf-8") as f:
        while True:
            inputs, targets=[],[]
            for _ in range(num_devices):
                d=train_data_stream.readline()
                if d == "": train_data_stream.seek(0); d=train_data_stream.readline()
                inp, tar= d.split("\t")
                inputs.append(np.asarray(np.pad(sp.encode(inp), (0, max_len))), dtype=np.int32)
                targets.append(np.asarray(np.pad(sp.encode(tar), (0, max_len))), dtype=np.int32)
            inputs = np.stack(inputs)
            targets = np.stack(targets)
            yield inputs, targets