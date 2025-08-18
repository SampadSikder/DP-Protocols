import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import math
from scipy.stats import binom, chisquare
from cfo.oue import OUE
from cfo.olh import OLH
from attacks.APA import AdaptivePatternAttack



def demonstrate_protocol_agnostic_attack():
    """Demonstrate protocol-agnostic APA attack across OUE, OLH"""
    
    # Parameters
    d = 10
    epsilon = 1.0
    n_users = 1000
    target_item = 3
    m_fake_users = 200
    
    print("=== Protocol-Agnostic APA Attack Demonstration ===")
    
    # Test different protocols
    protocols_to_test = [
        ('OUE', OUE(d, epsilon)),
        ('OLH', OLH(d, epsilon))
    ]
    
    results = {}
    
    # Generate common synthetic data
    item_probs = np.ones(d) * 0.12
    item_probs[target_item] = 0.04
    item_probs /= item_probs.sum()
    true_items = np.random.choice(d, n_users, p=item_probs)
    true_counts = np.bincount(true_items, minlength=d)
    
    print(f"\nOriginal true counts: {true_counts}")
    print(f"Target item {target_item} original count: {true_counts[target_item]}")
    
    fig, axes = plt.subplots(2, len(protocols_to_test), figsize=(5*len(protocols_to_test), 10))
    
    for idx, (protocol_name, ldp_mechanism) in enumerate(protocols_to_test):
        print(f"\n{'='*50}")
        print(f"Testing Protocol: {protocol_name}")
        print('='*50)
        
        legitimate_privatized = [ldp_mechanism.privatize(item) for item in true_items]
        
        # Estimate counts without attack
        clean_estimates = ldp_mechanism.estimate_counts(legitimate_privatized)
        
        # Initialize and execute protocol-agnostic APA attack
        apa = AdaptivePatternAttack(
            ldp_mechanism=ldp_mechanism,
            target_item=target_item,
            m_fake_users=m_fake_users,
            protocol_type=protocol_name
        )
        attacked_data = apa.execute_attack(legitimate_privatized)
        
        # Estimate counts with attack
        attacked_estimates = ldp_mechanism.estimate_counts(attacked_data)
        
        # Calculate attack effectiveness
        boost_amount = attacked_estimates[target_item] - clean_estimates[target_item]
        boost_percentage = (boost_amount / max(clean_estimates[target_item], 1)) * 100
        
        results[protocol_name] = {
            'clean_estimate': clean_estimates[target_item],
            'attacked_estimate': attacked_estimates[target_item],
            'boost_amount': boost_amount,
            'boost_percentage': boost_percentage,
            'clean_estimates_all': clean_estimates,
            'attacked_estimates_all': attacked_estimates
        }
        
        print(f"Clean estimate for target item: {clean_estimates[target_item]:.2f}")
        print(f"Attacked estimate for target item: {attacked_estimates[target_item]:.2f}")
        print(f"Boost amount: {boost_amount:.2f}")
        print(f"Boost percentage: {boost_percentage:.1f}%")
        
        # Plot results for this protocol
        # Top plot: Count estimates comparison
        x = range(d)
        width = 0.25
        axes[0, idx].bar([i-width for i in x], true_counts, width=width, 
                        label='True counts', alpha=0.8)
        axes[0, idx].bar(x, clean_estimates, width=width, 
                        label='Clean estimates', alpha=0.8)
        axes[0, idx].bar([i+width for i in x], attacked_estimates, width=width, 
                        label='Attacked estimates', alpha=0.8)
        axes[0, idx].axvline(x=target_item, color='red', linestyle='--', 
                           alpha=0.5, label=f'Target item {target_item}')
        axes[0, idx].set_xlabel('Item')
        axes[0, idx].set_ylabel('Count')
        axes[0, idx].set_title(f'{protocol_name}: Count Estimates')
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)
        
        # Bottom plot: Target item focus
        categories = ['True', 'Clean Est.', 'Attacked Est.']
        values = [true_counts[target_item], clean_estimates[target_item], attacked_estimates[target_item]]
        colors = ['blue', 'green', 'red']
        axes[1, idx].bar(categories, values, color=colors, alpha=0.7)
        axes[1, idx].set_ylabel('Count')
        axes[1, idx].set_title(f'{protocol_name}: Target Item {target_item}')
        for i, v in enumerate(values):
            axes[1, idx].text(i, v + 0.01 * max(values), f'{v:.1f}', ha='center', va='bottom')
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("PROTOCOL COMPARISON SUMMARY")
    print('='*70)
    print(f"{'Protocol':<15} {'Clean Est.':<12} {'Attacked Est.':<15} {'Boost':<10} {'Boost %':<10}")
    print('-'*70)
    
    for protocol_name, result in results.items():
        print(f"{protocol_name:<15} {result['clean_estimate']:<12.2f} "
              f"{result['attacked_estimate']:<15.2f} {result['boost_amount']:<10.2f} "
              f"{result['boost_percentage']:<10.1f}%")
    
    return results


def analyze_attack_detectability():
    """Analyze the detectability of APA attacks across protocols"""
    
    print("\n=== Attack Detectability Analysis ===")
    
    d = 10
    epsilon = 1.0
    n_users = 1000
    target_item = 3
    m_fake_users = 200
    
    protocols = [
        ('OUE', OUE(d, epsilon)),
        ('OLH', OLH(d, epsilon)),
    ]
    
    # Generate data
    item_probs = np.ones(d) * 0.12
    item_probs[target_item] = 0.04
    item_probs /= item_probs.sum()
    true_items = np.random.choice(d, n_users, p=item_probs)
    
    detection_results = {}
    
    for protocol_name, mechanism in protocols:
        print(f"\n--- {protocol_name} Detection Analysis ---")
        
        # Create legitimate and attacked datasets
        legitimate_data = [mechanism.privatize(item) for item in true_items]
        
        apa = AdaptivePatternAttack(mechanism, target_item, m_fake_users, protocol_name)
        attacked_data = apa.execute_attack(legitimate_data)
        
        # Statistical tests for detection
        if protocol_name == 'OUE':
            # Test distribution of ones
            legit_ones = [np.sum(report) for report in legitimate_data]
            attack_ones = [np.sum(report) for report in attacked_data]
            
            # Expected distribution based on OUE parameters
            p_binomial = (1/d) * (mechanism.p + (d-1) * mechanism.q)
            expected_probs = binom.pmf(range(d+1), d, p_binomial)
            
            # Chi-square test with proper normalization
            legit_hist = np.bincount(legit_ones, minlength=d+1)
            attack_hist = np.bincount(attack_ones, minlength=d+1)
            
            # Calculate expected counts
            expected_legit = expected_probs * len(legitimate_data)
            expected_attack = expected_probs * len(attacked_data)
            
            # Only test bins with sufficient expected count (>=5)
            mask_legit = expected_legit >= 5
            mask_attack = expected_attack >= 5
            
            try:
                if np.sum(mask_legit) > 1:
                    # Normalize to ensure sums match exactly
                    obs_legit = legit_hist[mask_legit].astype(float)
                    exp_legit = expected_legit[mask_legit]
                    
                    # Renormalize expected to match observed sum exactly
                    exp_legit = exp_legit * np.sum(obs_legit) / np.sum(exp_legit)
                    
                    chi2_legit, p_val_legit = chisquare(obs_legit, exp_legit)
                    print(f"Legitimate data chi-square: stat={chi2_legit:.3f}, p-value={p_val_legit:.3f}")
                else:
                    chi2_legit, p_val_legit = None, None
                    print("Insufficient data for legitimate chi-square test")
                
                if np.sum(mask_attack) > 1:
                    # Normalize to ensure sums match exactly
                    obs_attack = attack_hist[mask_attack].astype(float)
                    exp_attack = expected_attack[mask_attack]
                    
                    # Renormalize expected to match observed sum exactly
                    exp_attack = exp_attack * np.sum(obs_attack) / np.sum(exp_attack)
                    
                    chi2_attack, p_val_attack = chisquare(obs_attack, exp_attack)
                    print(f"Attacked data chi-square: stat={chi2_attack:.3f}, p-value={p_val_attack:.3f}")
                else:
                    chi2_attack, p_val_attack = None, None
                    print("Insufficient data for attacked chi-square test")
                    
                detection_results[protocol_name] = {
                    'legit_p_value': p_val_legit,
                    'attack_p_value': p_val_attack,
                    'suspicious': p_val_attack < 0.05 if p_val_attack is not None else False
                }
                
            except Exception as e:
                print(f"Chi-square test failed: {e}")
                detection_results[protocol_name] = {
                    'legit_p_value': None,
                    'attack_p_value': None,
                    'suspicious': False
                }
        
        
        elif protocol_name == 'OLH':
            # For OLH, analyze bucket distribution
            legit_buckets = [report['y'] for report in legitimate_data]
            attack_buckets = [report['y'] for report in attacked_data]
            
            legit_hist = np.bincount(legit_buckets, minlength=mechanism.g)
            attack_hist = np.bincount(attack_buckets, minlength=mechanism.g)
            
            # Expected uniform distribution across buckets
            expected_legit = np.full(mechanism.g, len(legitimate_data) / mechanism.g, dtype=float)
            expected_attack = np.full(mechanism.g, len(attacked_data) / mechanism.g, dtype=float)
            
            try:
                chi2_legit, p_val_legit = chisquare(legit_hist.astype(float), expected_legit)
                chi2_attack, p_val_attack = chisquare(attack_hist.astype(float), expected_attack)
                
                print(f"Legitimate data chi-square: stat={chi2_legit:.3f}, p-value={p_val_legit:.3f}")
                print(f"Attacked data chi-square: stat={chi2_attack:.3f}, p-value={p_val_attack:.3f}")
                
                detection_results[protocol_name] = {
                    'legit_p_value': p_val_legit,
                    'attack_p_value': p_val_attack,
                    'suspicious': p_val_attack < 0.05
                }
                
            except Exception as e:
                print(f"Chi-square test failed: {e}")
                detection_results[protocol_name] = {
                    'legit_p_value': None,
                    'attack_p_value': None,
                    'suspicious': False
                }
    
    # Summary of detectability
    print(f"\n{'='*50}")
    print("DETECTABILITY SUMMARY")
    print('='*50)
    print(f"{'Protocol':<10} {'Legit p-val':<12} {'Attack p-val':<12} {'Suspicious':<12}")
    print('-'*50)
    
    for protocol, result in detection_results.items():
        legit_p = result['legit_p_value']
        attack_p = result['attack_p_value']
        suspicious = result['suspicious']
        
        legit_str = f"{legit_p:.3f}" if legit_p is not None else "N/A"
        attack_str = f"{attack_p:.3f}" if attack_p is not None else "N/A"
        
        print(f"{protocol:<10} {legit_str:<12} {attack_str:<12} {suspicious}")
    
    return detection_results

def run_multiple_protocol_experiments(n_experiments=20):
    """Run multiple experiments across all protocols to analyze consistency"""
    
    print(f"\n=== Running {n_experiments} Experiments Across Protocols ===")
    
    d = 10
    epsilon = 1.0
    n_users = 1000
    target_item = 3
    m_fake_users = 200
    
    protocols = [
        ('OUE', lambda: OUE(d, epsilon)),
        ('OLH', lambda: OLH(d, epsilon))
    ]
    
    all_results = {name: {'boosts': [], 'percentages': []} for name, _ in protocols}
    
    for exp in range(n_experiments):
        if (exp + 1) % 5 == 0:
            print(f"Experiment {exp + 1}/{n_experiments}")
        
        # Generate common data for this experiment
        item_probs = np.ones(d) * 0.12
        item_probs[target_item] = 0.04
        item_probs /= item_probs.sum()
        true_items = np.random.choice(d, n_users, p=item_probs)
        
        for protocol_name, mechanism_factory in protocols:
            mechanism = mechanism_factory()
            
            # Create legitimate data
            legitimate_data = [mechanism.privatize(item) for item in true_items]
            clean_estimates = mechanism.estimate_counts(legitimate_data)
            
            # Execute attack
            apa = AdaptivePatternAttack(mechanism, target_item, m_fake_users, protocol_name)
            attacked_data = apa.execute_attack(legitimate_data)
            attacked_estimates = mechanism.estimate_counts(attacked_data)
            
            # Calculate metrics
            boost = attacked_estimates[target_item] - clean_estimates[target_item]
            boost_pct = (boost / max(clean_estimates[target_item], 1)) * 100
            
            all_results[protocol_name]['boosts'].append(boost)
            all_results[protocol_name]['percentages'].append(boost_pct)
    
    # Analyze results
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY ({n_experiments} runs)")
    print('='*70)
    print(f"{'Protocol':<10} {'Mean Boost':<12} {'Std Boost':<12} {'Mean %':<12} {'Std %':<10}")
    print('-'*70)
    
    for protocol_name in all_results:
        boosts = np.array(all_results[protocol_name]['boosts'])
        percentages = np.array(all_results[protocol_name]['percentages'])
        
        print(f"{protocol_name:<10} {np.mean(boosts):<12.2f} {np.std(boosts):<12.2f} "
              f"{np.mean(percentages):<12.1f} {np.std(percentages):<10.1f}")
    
    return all_results

if __name__ == "__main__":
    print("=== Protocol-Agnostic Adaptive Pattern Attack Demo ===")
    
    main_results = demonstrate_protocol_agnostic_attack()
    
    detection_results = analyze_attack_detectability()
    
    #experiment_results = run_multiple_protocol_experiments(n_experiments=10)