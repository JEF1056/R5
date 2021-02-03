import io
import os
import json
import numpy as np
import sentencepiece as spm
from random import randrange

with open("config.json", "r") as f:
    nq_tsv_path, max_len, out_dir=json.load(f)
sp = spm.SentencePieceProcessor(model_file=f"{os.path.join(out_dir,'bpe')}.model") 

def stream(num_devices, split, debug=False):
    with io.open(nq_tsv_path[split], mode="r", encoding="utf-8") as f:
        print("~~Getting offsets~~")
        line_offset, offset,line = [],0,None
        while line != "":
            line=f.readline()
            line_offset.append(offset)
            offset += len(line)
        f.seek(0)
        np.random.shuffle(line_offset)
        
        print(f"~~Initialized {split} stream~~")
        curr_index=0
        while True:
            inputs, mask=[],[]
            while len(inputs) < num_devices:
                f.seek(line_offset[curr_index])
                d=f.readline()
                curr_index+=1
                if curr_index >= len(line_offset)-1:  np.random.shuffle(line_offset)
                inp, tar= d.split("\t")
                inp, tar= sp.encode(inp), sp.encode(tar)
                if len(inp) < max_len or len(tar) < max_len:
                    combined=inp+[1]+tar+[2]
                    inputs.append(np.asarray(np.pad(combined, (0, max_len-len(combined))), dtype=np.int32))
                    mask.append(np.asarray(np.pad(np.ones(len(tar)+1), (len(inp)+1, max_len-len(combined))), dtype=np.int32))
            if debug:
                print(f"{len(inputs[0])}\n{len(mask[0])}")
                print(f"{inputs[0]}\n{mask[0]}\n")
                ovloc=sorted(np.where(mask[0]==1))
                st=ovloc[0][0]-10 if ovloc[0][0]-10 >= 0 else 0
                print(sp.decode(inputs[0].tolist()[st:ovloc[0][0]+10]))
                overlay = [int(inputs[0][i]) for i in ovloc[0]]
                print(f"Overlay: {overlay}\nDecoded: {sp.decode(overlay)}")
            inputs = np.stack(inputs)
            mask = np.stack(mask)
            yield inputs, inputs, mask