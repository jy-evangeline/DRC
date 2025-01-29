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
from berk_jones import berk_jones
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
                adjusted_scores = [detoxify_human_scores[idx] for idx, _ in C_lambda_]
                r_lambda_ = max(adjusted_scores)
                r_lambdas.append(r_lambda_)
        
        assert len(r_lambdas) == len(x_cal)
        
        var_r_lambda = np.percentile(r_lambdas,beta * 100)
        # empirical_risk = np.mean(r_lambdas)
        empirical_risk = np.mean([r for r in r_lambdas if r > var_r_lambda])
        max_values = np.maximum(r_lambdas, var_r_lambda)

        sigma_lambda = 1/(1-beta)*np.std(max_values)
        risks.append(empirical_risk+1.645*sigma_lambda/np.sqrt(len(r_lambdas)))

    risks  = np.array(risks)
    valid_lambdas = lambda_candidates[risks <= alpha]
    if valid_lambdas.size > 0:
        lambda_optimal = np.max(valid_lambdas)
    else:
        lambda_optimal = None
    return lambda_optimal


def distortion_risk_control_ltt(x_cal, y_cal, alpha, beta):
    lambda_candidates = np.linspace(0.0210, 0.8560, 1000)  # Range of lambda values for tuning
    risks = []
    # C_sets = defaultdict(dict)


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
                adjusted_scores = [detoxify_human_scores[idx] for idx, _ in C_lambda_]
                r_lambda_ = max(adjusted_scores)
                r_lambdas.append(r_lambda_)
        
        assert len(r_lambdas) == len(x_cal)
        
        var_r_lambda = np.percentile(r_lambdas,beta * 100)
        empirical_risk = np.mean([r for r in r_lambdas if r > var_r_lambda])
        max_values = np.maximum(r_lambdas, var_r_lambda)

        sigma_lambda = 1/(1-beta)*np.std(max_values)
        risks.append(empirical_risk+1.645*sigma_lambda/np.sqrt(len(r_lambdas)))

    risks  = np.array(risks)
    n_choose = len(risks)
    for i, risk in enumerate(risks):
        if risk > alpha:
            n_choose = i
            break
    if n_choose > 0:
        lambda_optimal = lambda_candidates[n_choose - 1]
    else:
        lambda_optimal = None
    return lambda_optimal

def distortion_risk_control_DKW(x_cal, y_cal, alpha, beta, n_samples):
    epsilon = np.sqrt(np.log(1 / 0.05) / (2 * n_samples))
    
    lambda_candidates = np.linspace(0.0210, 0.8560, 1000)  # 0.0035, 0.9745
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
                adjusted_scores = [detoxify_human_scores[idx] for idx, _ in C_lambda_]
                r_lambda_ = max(adjusted_scores)
                r_lambdas.append(r_lambda_)
        
        assert len(r_lambdas) == len(x_cal)
        n = len(r_lambdas)
        n_beta = min(int(np.ceil(n*(beta+epsilon)))-1,n-1)
        sorted_scores = np.sort(r_lambdas)
        empirical_cvar =  (n_beta/n-beta-epsilon)*sorted_scores[n_beta] + 1/n*np.sum([sorted_scores[i] for i in range(n_beta+1,n)])+epsilon*sorted_scores[-1]
        risks.append(empirical_cvar/(1-beta))

    valid_lambdas = lambda_candidates[np.array(risks) <= alpha]
    if valid_lambdas.size > 0:
        lambda_optimal = np.max(valid_lambdas)
    else:
        lambda_optimal = None

    return lambda_optimal

def distortion_risk_control_BJ(x_cal, y_cal, alpha, beta, n_samples):
    
    lambda_candidates = np.linspace(0.0210, 0.8560, 1000)  # 0.0035, 0.9745
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
                adjusted_scores = [detoxify_human_scores[idx] for idx, _ in C_lambda_]
                r_lambda_ = max(adjusted_scores)
                r_lambdas.append(r_lambda_)
        
        assert len(r_lambdas) == len(x_cal)
        n = len(r_lambdas)
        LB = berk_jones(n, 0.05)

        n_beta = n
        for i,item in enumerate(LB):
            if item >= beta:
                n_beta = i
                break
        sorted_scores = np.sort(r_lambdas)

        assert len(r_lambdas) == n, "Length of LB and r_lambdas must match."
        result = 0
        for i in range(n_beta,n):
            if i == 0:
                diff = LB[i] - beta
            else:
                diff = LB[i] - LB[i - 1]
            result += diff * sorted_scores[i]
    
        result += (1 - LB[-1]) * 1
        risks.append(result/(1-beta))

    try:
        valid_lambdas = lambda_candidates[np.array(risks) <= alpha]
        if valid_lambdas.size > 0:
            lambda_optimal = np.max(valid_lambdas)
        else:
            lambda_optimal = None
    except:
        lambda_optimal = None

    return lambda_optimal

def evaluate_remaining_data(remaining_x_cal, lambda_optimal, remaining_y_cal, beta):
    detoxify_ft_all = []
    detoxify_human_all = []
    sample_counts = []
    detoxify_human_all_max = []
    if not lambda_optimal:
        return None, None, None, None, None

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
    
    sorted_human_scores = np.sort(detoxify_human_all_max)
    threshold_index = int(np.ceil(beta * len(sorted_human_scores))) - 1
    beta_cvar_human_score = np.mean(sorted_human_scores[threshold_index:])
    
    mean_human_score = np.mean(detoxify_human_all)
    percentile_95_human_score = scoreatpercentile(detoxify_human_all, 95)
    max_human_score = np.max(detoxify_human_all)
    average_sample_count = np.mean(sample_counts) if sample_counts else 0

    return mean_human_score, percentile_95_human_score, max_human_score, beta_cvar_human_score, average_sample_count

def save_scores(scores, folder_name='score_results'):
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, 'scores.pkl'), 'wb') as f:
        pickle.dump(scores, f)

def main(trial_index, f1_score, use_dkw, use_bj,use_ltt,alpha,beta):
    directory = './results_llama2_7B_Real/test_toxic_new'
    conformal_directory = f'./results_llama2_7B_Real/conformal_set/conformal_set_size_F1_{f1_score}.pkl'
    
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
    corr, p_value = spearmanr(all_ft_scores, all_human_scores)

    print("Spearman Correlation Coefficient:", corr)
    print("P-value:", p_value)
    print(np.min(all_ft_scores),np.median(all_ft_scores),np.max(all_ft_scores))

    with open(conformal_directory, 'rb') as f:
        y_cal = pickle.load(f)
    
    key = 0 # use the responses with filter hyperparameter group 0
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
    elif use_ltt:
        lambda_optimal = distortion_risk_control_ltt(x_cal_train, y_cal_train, alpha, beta)
    else:
        lambda_optimal = distortion_risk_control(x_cal_train, y_cal_train, alpha, beta)
    
    if use_dkw:
        method_folder = "DKW"
    elif use_bj:
        method_folder = "BJ"
    elif use_ltt:
        method_folder = "DRC_LTT"
    else:
        method_folder = "DRC"

    base_folder = f'./results_detoxify_0.15/{method_folder}/trial_{trial_index}/alpha_{alpha}_beta_{beta}_{f1_score}'
    os.makedirs(base_folder, exist_ok=True)
    
    if lambda_optimal is not None:
        with open(os.path.join(base_folder, 'lambda_optimal.txt'), 'w') as f:
            f.write(f'Lambda Optimal: {lambda_optimal}\n')

    mean_human_score, percentile_95_human_score, max_human_score, beta_cvar_human_score, average_sample_count = evaluate_remaining_data(x_cal_remaining, lambda_optimal, y_cal_remaining,beta)

    scores = {
        'optimal_lambda':lambda_optimal,
        'mean_human_score': mean_human_score,
        'percentile_95_human_score': percentile_95_human_score,
        'max_human_score': max_human_score,
        'beta_cvar_human_score': beta_cvar_human_score,
        'average_sample_count': average_sample_count
    }
    save_scores(scores = scores, folder_name=base_folder)

    with open( os.path.join(base_folder,'results_summary.txt'), 'w') as f:
        f.write(f"Lambda Optimal: {lambda_optimal}\n")
        f.write(f"Mean Human Score: {mean_human_score}\n")
        f.write(f"95th Percentile Human Score: {percentile_95_human_score}\n")
        f.write(f"Max Human Score: {max_human_score}\n")
        f.write(f"Beta CVaR Human Score: {beta_cvar_human_score}\n")
        f.write(f"Average Sample Count: {average_sample_count}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_index", type=int, default=0, help="Index for trial to run")
    parser.add_argument("--f1_score", type=float, required=True, help="F1 score to use in filename")
    parser.add_argument("--use_dkw", type=bool, default=False, help="Whether to use the DKW approach")
    parser.add_argument("--use_bj", type=bool, default=False, help="Whether to use the Berk Jones approach")
    parser.add_argument("--use_ltt", type=bool, default=False, help="Whether to use the LTT approach")
    parser.add_argument("--alpha", type=float, default=0.35, help="Expected risk level")
    parser.add_argument("--beta", type=float, default=0.75, help="beta for CVaR")
    args = parser.parse_args()
    
    main(args.trial_index, args.f1_score, args.use_dkw, args.use_bj, args.use_ltt,args.alpha,args.beta)