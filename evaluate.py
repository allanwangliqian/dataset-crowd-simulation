import pickle
import argparse
import os

import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='group_simulator_evaluation')
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="exps/results",
        help="Path to evaluate experiment results"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    with open(os.path.join(args.output_dir, "results.pickle"), "rb") as fp:
        results = pickle.load(fp)
    num_cases = len(results)
    success = []
    navigation_time = []
    path_length = []
    path_smoothness = []
    motion_smoothness = []
    min_ped_distance = []
    avg_ped_distance = []
    state_time = []
    eval_time = []
    for i in range(num_cases):
        rst = results[i]['result']

        success.append(rst['success'])
        navigation_time.append(rst['navigation_time'])
        path_length.append(rst['path_length'])
        if not np.isnan(rst['path_smoothness']):
            path_smoothness.append(rst['path_smoothness'])
        motion_smoothness.append(rst['motion_smoothness'])
        min_ped_distance.append(rst['min_ped_dist'])
        avg_ped_distance.append(rst['avg_ped_dist'])
        state_time.append(rst['state_time'])
        eval_time.append(rst['eval_time'])

    success_rate = np.sum(success) / num_cases
    avg_navigation_time = np.mean(navigation_time)
    std_navigation_time = np.std(navigation_time)
    avg_path_length = np.mean(path_length)
    std_path_length = np.std(path_length)
    avg_path_smoothness = np.mean(path_smoothness)
    std_path_smoothness = np.std(path_smoothness)
    avg_motion_smoothness = np.mean(motion_smoothness)
    std_motion_smoothness = np.std(motion_smoothness)
    avg_min_ped_distance = np.mean(min_ped_distance)
    std_min_ped_distance = np.std(min_ped_distance)
    avg_avg_ped_distance = np.mean(avg_ped_distance)
    std_avg_ped_distance = np.std(avg_ped_distance)
    avg_state_time = np.mean(state_time)
    std_state_time = np.std(state_time)
    avg_eval_time = np.mean(eval_time)
    std_eval_time = np.std(eval_time)

    print("Success rate: {:.2f}".format(success_rate))
    print("Navigation time - mean: {:.2f} std: {:.2f}".format(avg_navigation_time, std_navigation_time))
    print("Path length - mean: {:.2f} std: {:.2f}".format(avg_path_length, std_path_length))
    print("Path smoothness - mean: {:.4f} std: {:.4f}".format(avg_path_smoothness, std_path_smoothness))
    print("Motion smoothness - mean: {:.2f} std: {:.2f}".format(avg_motion_smoothness, std_motion_smoothness))
    print("Min ped distance - mean: {:.2f} std: {:.2f}".format(avg_min_ped_distance, std_min_ped_distance))
    print("Avg ped distance - mean: {:.2f} std: {:.2f}".format(avg_avg_ped_distance, std_avg_ped_distance))
    print("State time - mean: {:.4f} std: {:.4f}".format(avg_state_time, std_state_time))
    print("Eval time - mean: {:.4f} std: {:.4f}".format(avg_eval_time, std_eval_time))
