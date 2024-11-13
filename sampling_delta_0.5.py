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


def VaR(scores, alpha):
    return scoreatpercentile(scores, 100 * (1 - alpha))

import numpy as np

def calculate_sigma_square(empirical_cdf, x_values):
    sigma_square = 0
    n = len(x_values)
    
    for i in range(n):
        for j in range(n):
            x_i = x_values[i]
            x_j = x_values[j]
            
            F_lambda_xi = empirical_cdf[x_i]
            F_lambda_xj = empirical_cdf[x_j]
            F_lambda_min_xi_xj = empirical_cdf[min(x_i, x_j)]
            
            sigma_square += (F_lambda_min_xi_xj - F_lambda_xi * F_lambda_xj)

    sigma_square /= (n * n)
    
    return sigma_square



def distortion_risk_control(x_cal, y_cal, alpha, delta):
    lambda_candidates = np.linspace(0.0210, 0.8560, 1000)  # Range of lambda values for tuning
    risks = []
    C_sets = defaultdict(dict)


    for lambda_ in tqdm(lambda_candidates):
        r_lambdas = []
        for key, val in x_cal.items():
            detoxify_ft_scores = val['detoxify_ft'][0].reshape(-1)
            detoxify_human_scores = val['detoxify_human']
            C_all = y_cal[key]
            C_lambda_ = [(idx,val) for idx, val in C_all if detoxify_ft_scores[idx] < lambda_]
            if len(C_lambda_) == 0:
                continue
            r_lambda_ = max([detoxify_human_scores[idx] for idx, _ in C_lambda_])
            r_lambdas.append(r_lambda_)
            C_sets[lambda_][key] = C_lambda_

        var_r_lambda = np.percentile(r_lambdas, (1 - delta) * 100)
        empirical_risk = np.mean([r for r in r_lambdas if r > var_r_lambda])
        max_values = np.maximum(r_lambdas, var_r_lambda)

        sigma_square_lambda = (1/(1-delta)**2)*np.std(max_values)**2
        risks.append(empirical_risk)

    risks  = np.array(risks)
    print(max(risks),min(risks))
    valid_lambdas = lambda_candidates[risks <= alpha]
    if valid_lambdas.size > 0:
        lambda_optimal = np.max(valid_lambdas)
    else:
        lambda_optimal = None
    return C_sets[lambda_optimal], lambda_optimal


def distortion_risk_control_DKW(x_cal, y_cal, alpha, delta, n_samples):
    # Calculate epsilon using DKW inequality
    epsilon = np.sqrt(np.log(1 / 0.5) / (2 * n_samples))
    
    lambda_candidates = np.linspace(0.0210, 0.8560, 1000)  # 0.0035, 0.9745
    risks = []
    C_sets = defaultdict(dict)
    
    for lambda_ in tqdm(lambda_candidates):
        r_lambdas = []
        for key, val in x_cal.items():
            detoxify_ft_scores = val['detoxify_ft'][0].reshape(-1)
            detoxify_human_scores = val['detoxify_human']
            C_all = y_cal[key]
            C_lambda_ = [(idx, val) for idx, val in C_all if detoxify_ft_scores[idx] < lambda_]
            if len(C_lambda_) == 0:
                continue
            adjusted_scores = [detoxify_human_scores[idx] for idx, _ in C_lambda_]
            r_lambda_ = max(adjusted_scores)
            r_lambdas.append(r_lambda_)
            C_sets[lambda_][key] = C_lambda_
        
        n = len(r_lambdas)
        n_beta = min(int(np.ceil(n*(1-delta+epsilon)))-1,n-1)
        # np.percentile(r_lambdas, 1-delta+epsilon)
        sorted_scores = np.sort(r_lambdas)
        empirical_cvar =  (n_beta/n-(1-delta)-epsilon)*sorted_scores[n_beta] + 1/n*np.sum([sorted_scores[i] for i in range(n_beta+1,n)])+epsilon*sorted_scores[-1]
        # empirical_cvar = np.mean(sorted_scores[n_beta:])/(1-delta)
        risks.append(empirical_cvar/delta)
        # sorted_data = np.sort(r_lambdas)
        # p = np.arange(1, n + 1) / n  # Empirical probabilities

        # # Step 2: Apply one-sided DKW to get the lower bound of the CDF
        # F_L = np.maximum(p - epsilon, 0)  # Lower bound CDF

        # # Step 3: Compute the inverse of the lower bound CDF, F^-1_L
        # F_L_inv = interp1d(F_L, sorted_data, bounds_error=False, fill_value=(sorted_data[0], sorted_data[-1]))

        # # Step 4: Integrate using F^-1_L for CVaR calculation
        # ps = np.linspace(1-delta, 1, 500)
        # F_L_inv_values = F_L_inv(ps)  

        # empirical_cvar = np.trapz(F_L_inv_values, ps) / (1 - delta)
        # risks.append(empirical_cvar)

    valid_lambdas = lambda_candidates[np.array(risks) <= alpha]
    if valid_lambdas.size > 0:
        lambda_optimal = np.max(valid_lambdas)
    else:
        lambda_optimal = None

    return C_sets[lambda_optimal], lambda_optimal

def evaluate_remaining_data(remaining_x_cal, lambda_optimal, remaining_y_cal, beta):
    detoxify_ft_all = []
    detoxify_human_all = []
    detoxify_human_all_max = []
    sample_counts = []

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
    #     C_lambda_selected = [(idx, val, detoxify_ft_scores[idx], detoxify_human_scores[idx]) for idx, val in C_all if detoxify_ft_scores[idx] <= lambda_optimal]
    #     C_lambda_non_selected = [(idx, val, detoxify_ft_scores[idx], detoxify_human_scores[idx]) for idx, val in C_all if detoxify_ft_scores[idx] > lambda_optimal]
    #     {}
    rows_selected_non_selected = []
    for key, val in remaining_x_cal.items():
        detoxify_ft_scores = val['detoxify_ft'][0].reshape(-1)
        detoxify_human_scores = val['detoxify_human']
        C_all = remaining_y_cal[key]

        # Select values based on the lambda_optimal threshold
        C_lambda_selected = [(idx, val, detoxify_ft_scores[idx], detoxify_human_scores[idx]) 
                            for idx, val in C_all if detoxify_ft_scores[idx] <= lambda_optimal]
        C_lambda_non_selected = [(idx, val, detoxify_ft_scores[idx], detoxify_human_scores[idx]) 
                             for idx, val in C_all if detoxify_ft_scores[idx] > lambda_optimal]
        # Combine all selected entries into one cell
        combined_selected = "; ".join([f"(idx: {item[0]}, val: {item[1]}, detoxify_ft: {item[2]}, detoxify_human: {item[3]})" for item in C_lambda_selected])
        
        combined_non_selected = "; ".join([f"(idx: {item[0]}, val: {item[1]}, detoxify_ft: {item[2]}, detoxify_human: {item[3]})" for item in C_lambda_non_selected])
    
        rows_selected_non_selected.append([key, combined_selected, combined_non_selected])

        detoxify_human_all.extend([item[3] for item in C_lambda_selected])
        try:
            detoxify_human_all_max.append(np.max([item[3] for item in C_lambda_selected]))
        except:
            pass
        detoxify_ft_all.extend([item[2] for item in C_lambda_selected])

    # Create DataFrame for the selected entries
    df_selected_combined = pd.DataFrame(rows_selected_non_selected, columns=['key', 'C_lambda_selected', 'C_lambda_excluded'])

    # Save the DataFrame to an Excel file
    output_selected_combined_path = "./detoxify_selected_combined_results.csv"
    df_selected_combined.to_csv(output_selected_combined_path, index=False)


    # Compute CVaR at beta level for detoxify_human_all
    # sorted_human_scores = np.sort(detoxify_human_all)
    # threshold_index = int(np.ceil(beta * len(sorted_human_scores))) - 1
    # beta_cvar_human_score = np.mean(sorted_human_scores[threshold_index:])

    sorted_human_scores = np.sort(detoxify_human_all_max)
    threshold_index = int(np.ceil(beta * len(sorted_human_scores))) - 1
    beta_cvar_human_score = np.mean(detoxify_human_all_max[threshold_index:])

    mean_human_score = np.mean(detoxify_human_all)
    percentile_95_human_score = scoreatpercentile(detoxify_human_all, 95)
    max_human_score = np.max(detoxify_human_all)
    average_sample_count = np.mean(sample_counts) if sample_counts else 0

    return mean_human_score, percentile_95_human_score, max_human_score, beta_cvar_human_score, average_sample_count

def save_scores(scores, folder_name='score_results',use_dkw=False):
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, 'scores.pkl'), 'wb') as f:
        pickle.dump(scores, f)

def main(n_trials, f1_score, use_dkw, alpha,delta):
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
    
    for key in y_cal:
        y_cal_current = y_cal[key]['set']
        lambda_optimal = None
        
        all_keys = list(x_cal.keys())
        random.seed(100)
        random.shuffle(all_keys)
        
        train_keys = all_keys[:6000]
        remaining_keys = all_keys[6000:]
        
        x_cal_train = {k: x_cal[k] for k in train_keys}
        y_cal_train = {k: y_cal_current[k]  for k in train_keys}
        x_cal_remaining = {k: x_cal[k] for k in remaining_keys}
        y_cal_remaining = {k: y_cal_current[k]  for k in remaining_keys}
        
        if use_dkw:
            C_set, lambda_optimal = distortion_risk_control_DKW(x_cal_train, y_cal_train, alpha, delta, len(train_keys))
        else:
            C_set, lambda_optimal = distortion_risk_control(x_cal_train, y_cal_train, alpha, delta)
        
        folder_name = f'./results_delta_0.5/results_key_{key}_trial_{n_trials}_score_{f1_score}_alpha_{alpha}_beta_{delta}'
        if use_dkw: 
            folder_name += '_use_dkw'
        os.makedirs(folder_name, exist_ok=True)
        if lambda_optimal is not None:
            with open(os.path.join(folder_name, 'lambda_optimal.txt'), 'w') as f:
                f.write(f'Lambda Optimal: {lambda_optimal}\n')
    
        mean_human_score, percentile_95_human_score, max_human_score, beta_cvar_human_score, average_sample_count = evaluate_remaining_data(x_cal_remaining, lambda_optimal, y_cal_remaining,1-delta)

        scores = {
            'mean_human_score': mean_human_score,
            'percentile_95_human_score': percentile_95_human_score,
            'max_human_score': max_human_score,
            'beta_cvar_human_score': beta_cvar_human_score,
            'average_sample_count': average_sample_count
        }
        save_scores(scores = scores, folder_name=folder_name, use_dkw = args.use_dkw)

        with open( os.path.join(folder_name,'results_summary.txt'), 'w') as f:
            f.write(f"Lambda Optimal: {lambda_optimal}\n")
            f.write(f"Mean Human Score: {mean_human_score}\n")
            f.write(f"95th Percentile Human Score: {percentile_95_human_score}\n")
            f.write(f"Max Human Score: {max_human_score}\n")
            f.write(f"Beta CVaR Human Score: {beta_cvar_human_score}\n")
            f.write(f"Average Sample Count: {average_sample_count}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--f1_score", type=float, required=True, help="F1 score to use in filename")
    parser.add_argument("--use_dkw", type=bool, default=False, help="Whether to use the DKW approach")
    parser.add_argument("--alpha", type=float, default=0.35, help="Whether to use the DKW approach")
    parser.add_argument("--delta", type=float, default=0.25, help="Whether to use the DKW approach")
    args = parser.parse_args()
    
    main(args.n_trials, args.f1_score, args.use_dkw,args.alpha,args.delta)