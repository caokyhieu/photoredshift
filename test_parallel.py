from concurrent.futures import ThreadPoolExecutor
e = ThreadPoolExecutor(6)
import time


# def slowadd(a, b, delay=1):
#     time.sleep(delay)
#     return a + b

# start = time.time()
# results = [slowadd(i, i, delay=1) for i in range(10)]
# print(f'Traditional time : {time.time()-start:.2f}')
# start = time.time()
# futures = [e.submit(slowadd, i, i, delay=1) for i in range(10)]
# results = [f.result() for f in futures]
# print(f'Parallel time : {time.time()-start:.2f}')


series = {}
for fn in filenames:   # Simple map over filenames
    series[fn] = pd.read_hdf(fn)['x']

results = {}

for a in filenames:    # Doubly nested loop over the same collection
    for b in filenames:  
        if a != b:     # Filter out bad elements
            results[a, b] = series[a].corr(series[b])  # Apply function

((a, b), corr) = max(results.items(), key=lambda kv: kv[1])  # Reduction