import numpy as np
from tqdm import tqdm
from scipy.stats import scoreatpercentile
from collections import defaultdict
import os
import pickle
import random
import argparse
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from berk_jones import berk_jones
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KernelDensity
import uuid

def distortion_risk_control(x_cal, y_cal, alpha, beta):
    lambda_candidates = np.linspace(0.0210, 0.8560, 1000)  # Range of lambda values for tuning
    risks = []

    for lambda_ in tqdm(lambda_candidates):
        r_lambdas = []
        for key, val in x_cal.items():
            detoxify_ft_scores = val['detoxify_ft'][0].reshape(-1)
            detoxify_human_scores = val['detoxify_human']
            C_all = y_cal[key]
            C_lambda_ = [(idx,val) for idx, val in C_all if detoxify_ft_scores[idx] <= lambda_]
            if len(C_lambda_) == 0:
                r_lambdas.append(0.0)
            else:
                r_lambda_ = max([detoxify_human_scores[idx] for idx, _ in C_lambda_])
                r_lambdas.append(r_lambda_)
            # C_sets[lambda_][key] = C_lambda_

        var_r_lambda = np.percentile(r_lambdas,beta*100)
        n_beta=int(np.ceil(beta * len(r_lambdas))) - 1

        def bootstrap_percentile(data, percent, B=1000):
            stats = []
            for _ in range(B):
                s = np.random.choice(data, len(data), replace=True)
                stats.append(np.percentile(s, percent))
            return np.var(stats, ddof=1)*B/(B-1)

        sigma_lambda = bootstrap_percentile(r_lambdas, beta*100, B=1000)
        risks.append(var_r_lambda+1.645*np.sqrt(sigma_lambda)/np.sqrt(len(r_lambdas)))

    risks  = np.array(risks)
    valid_lambdas = lambda_candidates[risks <= alpha]
    if valid_lambdas.size > 0:
        lambda_optimal = np.max(valid_lambdas)
    else:
        lambda_optimal = None
    return lambda_optimal


def distortion_risk_control_DKW(x_cal, y_cal, alpha, beta, n_samples):
    epsilon = np.sqrt(np.log(1 / 0.05) / (2 * n_samples))
    
    lambda_candidates = np.linspace(0.0210, 0.8560, 1000)  
    risks = []
    
    for lambda_ in tqdm(lambda_candidates):
        r_lambdas = []
        for key, val in x_cal.items():
            detoxify_ft_scores = val['detoxify_ft'][0].reshape(-1)
            detoxify_human_scores = val['detoxify_human']
            C_all = y_cal[key]
            C_lambda_ = [(idx, val) for idx, val in C_all if detoxify_ft_scores[idx] < lambda_]
            if len(C_lambda_) == 0:
                r_lambdas.append(0.0)
            else:
                r_lambda_ = max([detoxify_human_scores[idx] for idx, _ in C_lambda_])
                r_lambdas.append(r_lambda_)

        var_r_lambda = np.percentile(r_lambdas,(beta+epsilon) * 100)
        risks.append(var_r_lambda)

    valid_lambdas = lambda_candidates[np.array(risks) <= alpha]
    if valid_lambdas.size > 0:
        lambda_optimal = np.max(valid_lambdas)
    else:
        lambda_optimal = None

    return lambda_optimal

def distortion_risk_control_BJ(x_cal, y_cal, alpha, beta, n_samples):
    
    lambda_candidates = np.linspace(0.0210, 0.8560, 1000)  
    risks = []
    
    for lambda_ in tqdm(lambda_candidates):
        r_lambdas = []
        for key, val in tqdm(x_cal.items()):
            detoxify_ft_scores = val['detoxify_ft'][0].reshape(-1)
            detoxify_human_scores = val['detoxify_human']
            C_all = y_cal[key]
            C_lambda_ = [(idx, val) for idx, val in C_all if detoxify_ft_scores[idx] < lambda_]
            if len(C_lambda_) == 0:
                r_lambdas.append(0.0)
            else:
                r_lambda_ = max([detoxify_human_scores[idx] for idx, _ in C_lambda_])
                r_lambdas.append(r_lambda_)
            
        n = len(r_lambdas)
        LB = berk_jones(n, 0.05)

        n_beta = None
        for i,item in enumerate(LB):
            if item >= beta:
                n_beta = i
                break
        sorted_scores = np.sort(r_lambdas)
        if n_beta is None:
            risks.append(1)
        else:
            risks.append(sorted_scores[n_beta])

    valid_lambdas = lambda_candidates[np.array(risks) <= alpha]
    if valid_lambdas.size > 0:
        lambda_optimal = np.max(valid_lambdas)
    else:
        lambda_optimal = None

    return lambda_optimal

def evaluate_remaining_data(remaining_x_cal, lambda_optimal, remaining_y_cal, beta):
    detoxify_ft_all = []
    detoxify_human_all = []
    sample_counts = []
    detoxify_human_all_max = []

    for key, val in remaining_x_cal.items():
        detoxify_ft_scores = val['detoxify_ft'][0].reshape(-1)
        detoxify_human_scores = val['detoxify_human']
        C_all = remaining_y_cal[key]

        selected_human_scores = None
        sample_count = 0
        for idx, val in C_all:
            sample_count += 1
            if detoxify_ft_scores[idx] < lambda_optimal:
                sample_counts.append(sample_count)
                break
    rows_selected_non_selected = []
    for key, val in remaining_x_cal.items():
        detoxify_ft_scores = val['detoxify_ft'][0].reshape(-1)
        detoxify_human_scores = val['detoxify_human']
        C_all = remaining_y_cal[key]

        C_lambda_selected = [(idx, val, detoxify_ft_scores[idx], detoxify_human_scores[idx]) 
                            for idx, val in C_all if detoxify_ft_scores[idx] <= lambda_optimal]
        C_lambda_non_selected = [(idx, val, detoxify_ft_scores[idx], detoxify_human_scores[idx]) 
                             for idx, val in C_all if detoxify_ft_scores[idx] > lambda_optimal]
        combined_selected = "; ".join([f"(idx: {item[0]}, val: {item[1]}, detoxify_ft: {item[2]}, detoxify_human: {item[3]})" for item in C_lambda_selected])
        
        combined_non_selected = "; ".join([f"(idx: {item[0]}, val: {item[1]}, detoxify_ft: {item[2]}, detoxify_human: {item[3]})" for item in C_lambda_non_selected])
    
        rows_selected_non_selected.append([key, combined_selected, combined_non_selected])

        detoxify_human_all.extend([item[3] for item in C_lambda_selected])
        if C_lambda_selected:
            detoxify_human_all_max.append(np.max([item[3] for item in C_lambda_selected]))
        else:
            detoxify_human_all_max.append(0.0)
        detoxify_ft_all.extend([item[2] for item in C_lambda_non_selected])

    df_selected_combined = pd.DataFrame(rows_selected_non_selected, columns=['key', 'C_lambda_selected', 'C_lambda_excluded'])

    output_selected_combined_path = "./detoxify_selected_combined_results_var.csv"
    df_selected_combined.to_csv(output_selected_combined_path, index=False)
    
    sorted_human_scores = np.sort(detoxify_human_all_max)
    threshold_index = int(np.ceil(beta * len(sorted_human_scores))) - 1
    var_human_score = sorted_human_scores[threshold_index]
    
    mean_human_score = np.mean(detoxify_human_all)
    percentile_95_human_score = scoreatpercentile(detoxify_human_all, 95)
    max_human_score = np.max(detoxify_human_all)
    average_sample_count = np.mean(sample_counts) if sample_counts else 0

    return mean_human_score, percentile_95_human_score, max_human_score, var_human_score, average_sample_count

def save_scores(scores, folder_name='score_results'):
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, 'scores.pkl'), 'wb') as f:
        pickle.dump(scores, f)

def main(trial_index, f1_score, use_dkw, use_bj, alpha,beta):
    directory = './results_llama2_7B_Real/test_toxic_new'
    conformal_directory = f'./results_llama2_7B_Real/conformal_set/conformal_set_size_F1_{f1_score}.pkl'
    
    # Load X_cal and Y_cal data
    x_cal = {}
    all_ft_scores = []
    all_human_scores = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                for idx, entry in data.items():
                    key = idx
                    pred = entry['pred']
                    detoxify_ft_value = entry.get('detoxify_ft')
                    all_ft_scores.extend(detoxify_ft_value)
                    detoxify_human_value = entry.get('detoxify_human', {}).get('toxicity')
                    all_human_scores.extend(detoxify_human_value)
                    x_cal[key] = {
                        'pred': pred,
                        'detoxify_ft': detoxify_ft_value,
                        'detoxify_human': detoxify_human_value
                    }
    all_ft_scores = np.array(all_ft_scores).reshape(-1)
    all_human_scores = np.array(all_human_scores)
    with open('all_ft_scores.pkl','wb') as file:
        pickle.dump(all_ft_scores,file)
    with open('all_human_scores.pkl','wb') as file:
        pickle.dump(all_human_scores,file)
    corr, p_value = spearmanr(all_ft_scores, all_human_scores)

    print("Spearman Correlation Coefficient:", corr)
    print("P-value:", p_value)
    print(np.min(all_ft_scores),np.median(all_ft_scores),np.max(all_ft_scores))

    with open(conformal_directory, 'rb') as f:
        y_cal = pickle.load(f)
    
    key = 0
    y_cal_current = y_cal[key]['set']
    lambda_optimal = None
    
    all_keys = list(x_cal.keys())
    random.seed(int(uuid.uuid4().int % (2**32)) + trial_index)
    random.shuffle(all_keys)
    
    train_keys = all_keys[:6000]
    remaining_keys = all_keys[6000:]
    
    x_cal_train = {k: x_cal[k] for k in train_keys}
    y_cal_train = {k: y_cal_current[k]  for k in train_keys}
    x_cal_remaining = {k: x_cal[k] for k in remaining_keys}
    y_cal_remaining = {k: y_cal_current[k]  for k in remaining_keys}
    
    if use_dkw:
        lambda_optimal = distortion_risk_control_DKW(x_cal_train, y_cal_train, alpha, beta, len(train_keys))
    elif use_bj:
        lambda_optimal = distortion_risk_control_BJ(x_cal_train, y_cal_train, alpha, beta, len(train_keys))
    else:
        lambda_optimal = distortion_risk_control(x_cal_train, y_cal_train, alpha, beta)
    
    if use_dkw:
        method_folder = "DKW"
    elif use_bj:
        method_folder = "BJ"
    else:
        method_folder = "DRC"

    base_folder = f'./var_results_detoxify_0.15/{method_folder}/trial_{trial_index}/alpha_{alpha}_beta_{beta}_{f1_score}'
    os.makedirs(base_folder, exist_ok=True)
    
    if lambda_optimal is not None:
        with open(os.path.join(base_folder, 'lambda_optimal.txt'), 'w') as f:
            f.write(f'Lambda Optimal: {lambda_optimal}\n')

    mean_human_score, percentile_95_human_score, max_human_score, beta_var_human_score, average_sample_count = evaluate_remaining_data(x_cal_remaining, lambda_optimal, y_cal_remaining,beta)

    scores = {
        'optimal_lambda':lambda_optimal,
        'mean_human_score': mean_human_score,
        'percentile_95_human_score': percentile_95_human_score,
        'max_human_score': max_human_score,
        'var_human_score': beta_var_human_score,
        'average_sample_count': average_sample_count
    }
    save_scores(scores = scores, folder_name=base_folder)

    with open( os.path.join(base_folder,'results_summary.txt'), 'w') as f:
        f.write(f"Lambda Optimal: {lambda_optimal}\n")
        f.write(f"Mean Human Score: {mean_human_score}\n")
        f.write(f"95th Percentile Human Score: {percentile_95_human_score}\n")
        f.write(f"Max Human Score: {max_human_score}\n")
        f.write(f"VaR Human Score: {beta_var_human_score}\n")
        f.write(f"Average Sample Count: {average_sample_count}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_index", type=int, default=0, help="Number of trials to run")
    parser.add_argument("--f1_score", type=float, required=True, help="F1 score to use in filename")
    parser.add_argument("--use_dkw", type=bool, default=False, help="Whether to use the DKW approach")
    parser.add_argument("--use_bj", type=bool, default=False, help="Whether to use the Berk-Jones approach")
    parser.add_argument("--alpha", type=float, default=0.35, help="Risk control level")
    parser.add_argument("--beta", type=float, default=0.75, help="Percentile")
    args = parser.parse_args()
    
    main(args.trial_index, args.f1_score, args.use_dkw,args.use_bj, args.alpha,args.beta)