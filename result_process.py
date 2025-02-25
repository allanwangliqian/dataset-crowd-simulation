import pickle
import os
import numpy as np

def combine_results(results1_dir, results2_dir):
    """
    Combine results from different 2 runs
    """
    with open(os.path.join(results1_dir, "results.pickle"), "rb") as fp:
        results1 = pickle.load(fp)
    with open(os.path.join(results2_dir, "results.pickle"), "rb") as fp:
        results2 = pickle.load(fp)
    combined_results = results1 + results2
    return combined_results

def retry_results(results_dir):
    """
    Retry results
    """
    new_results = []
    with open(os.path.join(results_dir, "results.pickle"), "rb") as fp:
        results = pickle.load(fp)
    with open(os.path.join(results_dir, "retry-results.pickle"), "rb") as fp:
        retry_results = pickle.load(fp)

    counter = 0
    for result in results:
        rst = result['result']['success']
        if rst:
            new_results.append(result)
        else:
            new_results.append(retry_results[counter])
            counter += 1
            assert counter <= len(retry_results)

    return new_results