import gin
import trax
import os
import time
import argparse
import numpy as np
import sentencepiece as spm
import jax.numpy as jnp

parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument('-dir', type=str, default="train",
                    help='location of the model weights, tokenizer, and logs')
parser.add_argument('-backend', type=str, default="tensorflow-numpy",
                    help='backend to use for evaluation', choices=["jax", "tensorflow-numpy"])
parser.add_argument('-temp', type=float, default=1.0,
                    help='backend to use for evaluation')

print("~~Parsing Arguments~~")
args = parser.parse_args()
trax.fastmath.set_backend(args.backend)
gin.parse_config_file(os.path.join(args.dir, "config.gin"))

print("~~Loading Model~~")
model = trax.models.ReformerLM(mode="predict")
model.init_from_file(os.path.join(args.dir, "model.pkl.gz"),  weights_only=True)
model_init=model.state

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample(logits, top_k=0, top_p=0.0, repeat_filter=0.0, current_symbols=None, filter_value=-float('Inf')):
    logits = np.array(logits).astype(np.float64)
    sorted_indices = np.argsort(logits)[::-1]
    if top_k > 0:
        for i in range(top_k-1,len(sorted_indices)):
            logits[sorted_indices[i]]=filter_value
            
    if top_p > 0.0:
        cumulative_probs =np.cumsum(softmax(logits))
        for i, cprob in enumerate(cumulative_probs):
            if cprob > top_p:
                logits[sorted_indices[i]]=filter_value

    if repeat_filter > 0:
        assert current_symbols != None
        assert type(current_symbols) == list
        for i in current_symbols:
            logits[i] = logits[i] * (1-repeat_filter)
        
    return logits

sp = spm.SentencePieceProcessor(model_file=f"{os.path.join(args.dir,'bpe')}.model")

while True:
    inp=input("> ")
    inp=sp.encode(inp)+[1]
    print(sp.decode(inp))
    print(inp)
    #model.state=model_init
    #sampler = trax.supervised.decoding.autoregressive_sample_stream(model, inputs=inp, batch_size=1, temperature=args.temp, accelerate=True)
    current_symbols=[]
    s, p=time.time(),[]
    while len(current_symbols) < 30 and 2 not in current_symbols:
        print("started")
        t1=time.time()
        cont=jnp.asarray([current_symbols+inp], dtype=np.int32)
        print(cont)
        #next_token=next(sampler)
        output = model(cont)[:, -1, :][0] / args.temp
        filtered_logits=sample(output, top_k=4, top_p=0.9, repeat_filter=0.2, current_symbols=current_symbols)
        probabilities = softmax(filtered_logits)
        next_token = np.argmax(np.random.multinomial(1,probabilities, size=1)[0])
        current_symbols.append(int(next_token))
        print(sp.decode(current_symbols))
        p.append(time.time()-t1)
    e=time.time()
    print(sp.decode(current_symbols))
    print(current_symbols)
    print(f"Took {e-s} seconds {p}, avg: {sum(p)//len(p)}")