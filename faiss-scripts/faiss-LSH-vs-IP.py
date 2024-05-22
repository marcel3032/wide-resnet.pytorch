import numpy as np
import faiss
import time
import argparse


def lsh_search(batch_size, bits, unfold, out, k_similar):
    index = faiss.IndexLSH(batch_size, int(bits * batch_size))
    index.train(unfold)
    index.add(unfold)

    D, I = index.search(out, k_similar)


def ip_search(batch_size, bits, unfold, out, k_similar):
    index = faiss.IndexFlatL2(batch_size)
    index.add(unfold)

    D, I = index.search(out, k_similar)


def perform_search(methods, u_size, o_size, batch_size, bits, k_similar, repetitions=10):
    # conv.weight matrix = a*b*c*d
    # u_size = b*c*d
    # o_size = a
    
    out_size = u_size
    k_similar = int(out_size * k_similar)

    time_sum = {name: 0 for name in methods}
    for i in range(repetitions):
        unfold = np.random.random((u_size, batch_size)).astype('float32')
        out = np.random.random((o_size, batch_size)).astype('float32')
        for name in methods:
            ts = time.time()
            methods[name](batch_size, bits, unfold, out, k_similar)
            te = time.time()

            time_sum[name] += (te - ts)

    for name in time_sum:
        time_sum[name] /= repetitions
    
    return time_sum

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--bits', type=float)
parser.add_argument('--k_similar', type=float)

args = parser.parse_args()

print(args)

np.random.seed(int((12345+int(args.batch_size))*int(args.bits)**int(args.k_similar)))

sizes = [8, 16, 32, 64, 128, 256 , 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

ip_table = []
lsh_table = []
for u_size in sizes:
    print(f"u_size: {u_size}")
    ip_table.append([])
    lsh_table.append([])
    for o_size in sizes:
        print(f"  o_size: {o_size}", flush=True)
        res = perform_search({"ip": ip_search, "lsh": lsh_search}, u_size=u_size, o_size=o_size, batch_size=args.batch_size, bits=args.bits, k_similar=args.k_similar)
        ip_table[-1].append(res["ip"])
        lsh_table[-1].append(res["lsh"])

print("ip")
for r in ip_table:
    for c in r:
        print(f"{c:>8.5f}", end='')
    print()

print("lsh")
for r in lsh_table:
    for c in r:
        print(f"{c:>8.5f}", end='')
    print()

print("lsh<ip")
for r in range(len(lsh_table)):
    for c in range(len(lsh_table[r])):
        print(f"{lsh_table[r][c]<ip_table[r][c]:>3}", end='')
    print()

with open(f"batch_size_{args.batch_size}_bits_{args.bits}_k_similar_{args.k_similar}".replace(".", "_")+".py", 'w') as fout:
  print("ip =", ip_table, file=fout)
  print()
  print("lsh =", lsh_table, file=fout)

