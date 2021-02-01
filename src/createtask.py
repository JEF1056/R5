import io
import os
import json
import numpy as np
import sentencepiece as spm

with open("config.json", "r") as f:
    nq_tsv_path, max_len, out_dir=json.load(f)
sp = spm.SentencePieceProcessor(model_file=f"{os.path.join(out_dir,'bpe')}.model")

def stream(num_devices, split, debug=False):
    with io.open(nq_tsv_path[split], mode="r", encoding="utf-8") as f:
        print(f"~~Initialized {split} stream~~")
        while True:
            inputs, mask=[],[]
            while len(inputs) < num_devices:
                d=f.readline()
                if d == "": f.seek(0); d=f.readline()
                inp, tar= d.split("\t")
                inp, tar= sp.encode(inp), sp.encode(tar)
                if len(inp) < max_len or len(tar) < max_len:
                    combined=inp+[1]+tar+[2]
                    inputs.append(np.asarray(np.pad(combined, (0, max_len-len(combined))), dtype=np.int32))
                    mask.append(np.asarray(np.pad(np.ones(len(tar)+1), (len(inp)+1, max_len-len(combined))), dtype=np.int32))
            if debug:
                print(f"{len(inputs[0])}\n{len(mask[0])}")
                print(f"{inputs[0].tolist()}\n{mask[0].tolist()}")
                print()
                print(sp.decode(inputs[0].tolist()))
                overlay=[v for i, v in enumerate(inputs[0]) if mask[0][i] == 1]
                print(f"Overlay: {overlay}\nDecoded: {sp.decode(overlay)}")
                exit()
            inputs = np.stack(inputs)
            mask = np.stack(mask)
            yield inputs, inputs, mask