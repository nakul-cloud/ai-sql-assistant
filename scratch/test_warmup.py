import time
import torch
from FlagEmbedding import BGEM3FlagModel

print("Setting torch threads to 4...")
torch.set_num_threads(4)

print("Loading BGEM3FlagModel...")
start = time.time()
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print(f"Model loaded in {time.time() - start:.2f}s")

# Test 1: First inference without warmup
print("Running first encoding (warmup)...")
start = time.time()
model.encode(["warmup"], return_dense=True, return_sparse=True, return_colbert_vecs=False)
print(f"First encoding completed in {time.time() - start:.2f}s")

# Test 2: Second inference
print("Running second encoding (actual query)...")
start = time.time()
res = model.encode(["Show employees from Engineering"], return_dense=True, return_sparse=True, return_colbert_vecs=False)
print(f"Second encoding completed in {time.time() - start:.2f}s")

# Test 3: Third inference
print("Running third encoding (another query)...")
start = time.time()
res2 = model.encode(["List departments and manager names"], return_dense=True, return_sparse=True, return_colbert_vecs=False)
print(f"Third encoding completed in {time.time() - start:.2f}s")
