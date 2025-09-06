import pickle
import argparse
import os
import json

import numpy as np
import pandas as pd

from sim.data_loader import DataLoader

DL_ZARA2 = DataLoader('ucy', 1, base_path='sim')
DL_UNIV = DataLoader('ucy', 2, base_path='sim')

def get_args():
    parser = argparse.ArgumentParser(description='group_simulator_evaluation')
    parser.add_argument(
        "--case-file", 
        type=str, 
        default="data/all.json",
        help="Path to get case information"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="exps/results",
        help="Path to evaluate experiment results"
    )
    args = parser.parse_args()
    return args

def eq_pos(pos1, pos2):
    epsilon = 1e-5
    return (abs(pos1[0] - pos2[0]) < epsilon) and (abs(pos1[1] - pos2[1]) < epsilon)

def case_in_dataset(positions, dset):
    if dset == 'zara2':
        DL = DL_ZARA2
    elif dset == 'univ':
        DL = DL_UNIV
    else:
        raise ValueError(f"Unloaded or unknown dataset: {dset}")
    
    num_ppl = len(positions)
    all_ppl = DL.video_position_matrix
    is_in_dset = False
    for people in all_ppl:
        if len(people) == num_ppl:
            is_in_dset = True
            for pos in positions:
                maybe_in_dset = False
                for person in people:
                    if eq_pos(pos, person):
                        maybe_in_dset = True
                        break
                if not maybe_in_dset:
                    is_in_dset = False
                    break
            if is_in_dset:
                break
    return is_in_dset

def case_from_start_and_idx(hist):
    start_pos = hist[0]['robot_pos']
    ped_positions = hist[0]['pedestrians_pos']
    case_info = {'case': None, 'task': None}

    if (eq_pos(start_pos, [-8,5])) or (eq_pos(start_pos, [15,5])):
        case_info['case'] = 'eth'
        case_info['task'] = 'cross'
    if (eq_pos(start_pos, [5,0])) or (eq_pos(start_pos, [5,12.5])):
        case_info['case'] = 'eth'
        case_info['task'] = 'follow'

    if (eq_pos(start_pos, [2,-10.5])) or (eq_pos(start_pos, [2,4.5])):
        case_info['case'] = 'hotel'
        case_info['task'] = 'cross'
    if (eq_pos(start_pos, [5,-3])) or (eq_pos(start_pos, [-3,-3])):
        case_info['case'] = 'hotel'
        case_info['task'] = 'follow'

    if (eq_pos(start_pos, [-0.5,5])) or (eq_pos(start_pos, [15.5,5])):
        case_info['case'] = 'zara1'
        case_info['task'] = 'cross'
    if (eq_pos(start_pos, [7.5,8])) or (eq_pos(start_pos, [7.5,2])):
        case_info['case'] = 'zara1'
        case_info['task'] = 'follow'

    if (eq_pos(start_pos, [-0.5,6])) or (eq_pos(start_pos, [16,6]) and case_in_dataset(ped_positions, 'zara2')):
        case_info['case'] = 'zara2'
        case_info['task'] = 'cross'
    if (eq_pos(start_pos, [7.5,9])) or (eq_pos(start_pos, [7.5,2.5])):
        case_info['case'] = 'zara2'
        case_info['task'] = 'follow'

    if (eq_pos(start_pos, [0,6])) or (eq_pos(start_pos, [16,6]) and case_in_dataset(ped_positions, 'univ')):
        case_info['case'] = 'univ'
        case_info['task'] = 'cross'
    if (eq_pos(start_pos, [7.5,13])) or (eq_pos(start_pos, [7.5,0])):
        case_info['case'] = 'univ'
        case_info['task'] = 'follow'

    if case_info['case'] is None or case_info['task'] is None:
        raise ValueError("Cannot identify case and task from start position: {} and pedestrian positions: {}".format(start_pos, ped_positions))

    return case_info['case'] + '_' + case_info['task']

def init_case(rst):
    case_dict = {
        'success': [rst['success']],
        'navigation_time': [rst['navigation_time']],
        'path_length': [rst['path_length']],
        'path_smoothness': [rst['path_smoothness']],
        'motion_smoothness': [rst['motion_smoothness']],
        'min_ped_distance': [rst['min_ped_dist']],
        'avg_ped_distance': [rst['avg_ped_dist']],
        'state_time': [rst['state_time']],
        'eval_time': [rst['eval_time']],
    }
    return case_dict

def update_case(case_dict, rst):
    case_dict['success'].append(rst['success'])
    case_dict['navigation_time'].append(rst['navigation_time'])
    case_dict['path_length'].append(rst['path_length'])
    case_dict['path_smoothness'].append(rst['path_smoothness'])
    case_dict['motion_smoothness'].append(rst['motion_smoothness'])
    case_dict['min_ped_distance'].append(rst['min_ped_dist'])
    case_dict['avg_ped_distance'].append(rst['avg_ped_dist'])
    case_dict['state_time'].append(rst['state_time'])
    case_dict['eval_time'].append(rst['eval_time'])
    return case_dict

def summarize_case(case):
    case_summary = {
        "success-mean": np.mean(case['success']),
        "navigation_time-mean": np.mean(case['navigation_time']),
        "path_length-mean": np.mean(case['path_length']),
        "path_smoothness-mean": np.nanmean(case['path_smoothness']),
        "motion_smoothness-mean": np.mean(case['motion_smoothness']),
        "min_ped_distance-mean": np.mean(case['min_ped_distance']),
        "avg_ped_distance-mean": np.mean(case['avg_ped_distance']),
        "state_time-mean": np.mean(case['state_time']),
        "eval_time-mean": np.mean(case['eval_time']),
        "success-std": np.std(case['success']),
        "navigation_time-std": np.std(case['navigation_time']),
        "path_length-std": np.std(case['path_length']),
        "path_smoothness-std": np.nanstd(case['path_smoothness']),
        "motion_smoothness-std": np.std(case['motion_smoothness']),
        "min_ped_distance-std": np.std(case['min_ped_distance']),
        "avg_ped_distance-std": np.std(case['avg_ped_distance']),
        "state_time-std": np.std(case['state_time']),
        "eval_time-std": np.std(case['eval_time']),
    }
    return case_summary

if __name__ == "__main__":
    args = get_args()

    with open(args.case_file, "r") as f:
        case_info = json.load(f)

    with open(os.path.join(args.output_dir, "results.pickle"), "rb") as fp:
        results = pickle.load(fp)

    # if not len(case_info) == len(results):
    #     print(len(case_info), len(results))
    #     raise ValueError("Number of cases in case file does not match number of results.")

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

    print("--------------- Results Overview ---------------")
    print("Num cases: {}".format(num_cases))
    print("Success rate: {:.2f}".format(success_rate))
    print("Navigation time - mean: {:.2f} std: {:.2f}".format(avg_navigation_time, std_navigation_time))
    print("Path length - mean: {:.2f} std: {:.2f}".format(avg_path_length, std_path_length))
    print("Path smoothness - mean: {:.4f} std: {:.4f}".format(avg_path_smoothness, std_path_smoothness))
    print("Motion smoothness - mean: {:.2f} std: {:.2f}".format(avg_motion_smoothness, std_motion_smoothness))
    print("Min ped distance - mean: {:.2f} std: {:.2f}".format(avg_min_ped_distance, std_min_ped_distance))
    print("Avg ped distance - mean: {:.2f} std: {:.2f}".format(avg_avg_ped_distance, std_avg_ped_distance))
    print("State time - mean: {:.4f} std: {:.4f}".format(avg_state_time, std_state_time))
    print("Eval time - mean: {:.4f} std: {:.4f}".format(avg_eval_time, std_eval_time))

    results_dict = {}
    for i in range(num_cases):
        hist = results[i]['history']
        case = case_from_start_and_idx(hist)
        rst = results[i]['result']
        if not case in results_dict.keys():
            results_dict[case] = init_case(rst)
        else:
            results_dict[case] = update_case(results_dict[case], rst)

    df = pd.DataFrame.from_dict(results_dict, orient='index')
    df = df.reindex(["eth_cross", "hotel_cross", "zara1_cross", "zara2_cross", "univ_cross", "eth_follow", "hotel_follow", "zara1_follow", "zara2_follow", "univ_follow"])
    # df.to_csv(os.path.join(args.output_dir, "results.csv"))
    df.to_csv(os.path.join(args.output_dir, "results.csv"))

    results_summary_dict = {}
    for case in results_dict.keys():
        results_summary_dict[case] = summarize_case(results_dict[case])
    df_summary = pd.DataFrame.from_dict(results_summary_dict, orient='index')
    df_summary = df_summary.reindex(["eth_cross", "hotel_cross", "zara1_cross", "zara2_cross", "univ_cross", "eth_follow", "hotel_follow", "zara1_follow", "zara2_follow", "univ_follow"])
    df_summary.to_csv(os.path.join(args.output_dir, "results_summary.csv"))
