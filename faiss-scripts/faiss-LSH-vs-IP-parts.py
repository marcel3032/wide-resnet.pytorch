import numpy as np
import faiss
import time
import argparse

class MeasuredTime:
    def __init__(self):
        self.create_index = 0.0
        self.add_data = 0.0
        self.train_index = 0.0
        self.search = 0.0
    
    def __repr__(self):
        return f"create: {self.create_index:>2.5f} add: {self.add_data:>2.5f} train: {self.train_index:>2.5f} search: {self.search:>2.5f}"
    
    def __str__(self):
        return repr(self)
    
    def __add__(self, other):
        return MeasuredTime(self.create_index + other.create_index,
                            self.add_data + other.add_data,
                            self.train_index + other.train_index,
                            self.search + other.search)
    
    def __iadd__(self, other):
        self.create_index += other.create_index
        self.add_data += other.add_data
        self.train_index += other.train_index
        self.search += other.search
        return self
        
    def __truediv__(self, a):
        return MeasuredTime(self.create_index / a,
                            self.add_data / a,
                            self.train_index / a,
                            self.search / a)
    
    def __itruediv__(self, a):
        self.create_index /= a
        self.add_data /= a
        self.train_index /= a
        self.search /= a
        return self
        

def lsh_search(batch_size, bits, unfold, out, k_similar):
    res_time = MeasuredTime()
    
    # create
    ts = time.time()
    index = faiss.IndexLSH(batch_size, int(bits * batch_size))
    te = time.time()
    res_time.create_index = te-ts
    
    # train
    ts = time.time()
    index.train(unfold)
    te = time.time()
    res_time.train_index = te-ts
    
    # add
    ts = time.time()
    index.add(unfold)
    te = time.time()
    res_time.add_data = te-ts

    # search
    ts = time.time()
    D, I = index.search(out, k_similar)
    te = time.time()
    res_time.search = te-ts
    
    return res_time


def ip_search(batch_size, bits, unfold, out, k_similar):
    res_time = MeasuredTime()
    
    # create
    ts = time.time()
    index = faiss.IndexFlatL2(batch_size)
    te = time.time()
    res_time.create_index = te-ts
    
    # add
    ts = time.time()
    index.add(unfold)
    te = time.time()
    res_time.add_data = te-ts

    # search
    ts = time.time()
    D, I = index.search(out, k_similar)
    te = time.time()
    res_time.search = te-ts
    
    return res_time


def perform_search(methods, u_size, o_size, batch_size, bits, k_similar, repetitions=10):
    # conv.weight matrix = a*b*c*d
    # u_size = b*c*d
    # o_size = a
    
    out_size = u_size
    k_similar = int(out_size * k_similar)

    time_sum = {name: 0 for name in methods}
    time_parts = {name: MeasuredTime() for name in methods}
    for i in range(repetitions):
        unfold = np.random.random((u_size, batch_size)).astype('float32')
        out = np.random.random((o_size, batch_size)).astype('float32')
        for name in methods:
            ts = time.time()
            t = methods[name](batch_size, bits, unfold, out, k_similar)
            te = time.time()

            # print(time_parts[name], t, sep="\n", flush=True)
            # print()
            time_sum[name] += (te - ts)
            time_parts[name] += t

    for name in time_sum:
        time_sum[name] /= repetitions
        time_parts[name] /= repetitions
    
    return time_sum, time_parts

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--bits', type=float)
parser.add_argument('--k_similar', type=float)

args = parser.parse_args()

print(args)

np.random.seed(int((12345+int(args.batch_size))*int(args.bits)**int(args.k_similar)))

u_size = 8192*2
o_size = 8192*2
methods = {"ip": ip_search, "lsh": lsh_search}
# methods = {"ip": ip_search}
# methods = {"lsh": lsh_search}
times, time_parts = perform_search(methods, u_size=u_size, o_size=o_size, batch_size=args.batch_size, bits=args.bits, k_similar=args.k_similar)

print(times)
print()
print(time_parts)
