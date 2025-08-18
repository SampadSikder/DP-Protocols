import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import math
from scipy.stats import binom, chisquare

class AdaptivePatternAttack:
    """
    Adaptive Pattern Attack (APA) implementation for OUE
    
    This attack generates fake users with reports that follow the natural distribution
    pattern of the LDP protocol to avoid detection while manipulating frequency estimates.
    """
    
    def __init__(self, ldp_mechanism, target_item: int, m_fake_users: int, protocol_type: str):
        """
        Initialize APA attack
        
        Args:
            ldp_mechanism: LDP mechanism instance (OUE, OLH)
            target_item: Item to boost (0 to d-1)
            m_fake_users: Number of fake users to inject
            protocol_type: String specifying the protocol ('OUE', 'OLH') not for GRR
        """
        self.ldp_mechanism = ldp_mechanism
        self.target_item = target_item
        self.m_fake_users = m_fake_users
        self.protocol_type = protocol_type.upper()
        
        # Extract common parameters
        
    
        self.d = ldp_mechanism.d
        self.epsilon = ldp_mechanism.epsilon
        
        # Protocol-specific parameters
        self.protocol_params = self._get_protocol_params()
        
        # Construct omega distribution
        self.omega = self.construct_omega()
        
        print(f"\n=== Protocol-Agnostic Adaptive Pattern Attack Setup ===")
        print(f"Protocol: {self.protocol_type}")
        print(f"Target item: {target_item}")
        print(f"Number of fake users: {m_fake_users}")
        print(f"Protocol parameters: {self.protocol_params}")
        print(f"Omega distribution (first 10): {self.omega[:min(10, len(self.omega))]}")
        
    def _get_protocol_params(self) -> dict:
        """
        Extract protocol-specific parameters
        """
        params = {'protocol': self.protocol_type}
        
        if self.protocol_type == 'OUE':
            params['p'] = self.ldp_mechanism.p
            params['q'] = self.ldp_mechanism.q
        elif self.protocol_type == 'OLH':
            params['g'] = self.ldp_mechanism.g
            params['p'] = self.ldp_mechanism.p
            params['q'] = self.ldp_mechanism.q
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol_type}")
        
        return params
    
    def construct_omega(self) -> np.ndarray:
        if self.protocol_type == 'OUE':
            return self._construct_omega_oue()
        elif self.protocol_type == 'OLH':
            return self._construct_omega_olh()
        else:
            raise ValueError(f"Omega construction not implemented for {self.protocol_type}")
    
    def _construct_omega_oue(self) -> np.ndarray:
        p = self.protocol_params['p']
        q = self.protocol_params['q']
        
        # p_binomial represents the probability of getting a 1 in any position
        p_binomial = (1.0 / self.d) * (p + (self.d - 1) * q)
        
        # Generate theoretical distribution of number of ones
        k_values = np.arange(self.d + 1)
        theoretical_pdf = binom.pmf(k_values, self.d, p_binomial)
        theoretical_pdf /= theoretical_pdf.sum()  # Normalize
        
        # Distribute fake users according to this distribution
        omega = np.round(theoretical_pdf * self.m_fake_users).astype(int)
        
        # Ensure total fake users equals m_fake_users
        diff = self.m_fake_users - np.sum(omega)
        if diff != 0:
            max_idx = np.argmax(omega)
            omega[max_idx] += diff
        
        return omega
    
    
    
    def _construct_omega_olh(self) -> np.ndarray:
        g = self.protocol_params['g']
        p = self.protocol_params['p']
        q = self.protocol_params['q']
        
       
        p_binomial = (1.0 / self.d) * (p + (self.d - 1) * (q + (p - q) / g))
        
        # Use simpler distribution for OLH - focus on bucket matches
        # Create distribution based on expected matches per report
        max_patterns = min(self.d, g) + 1
        k_values = np.arange(max_patterns)
        theoretical_pdf = binom.pmf(k_values, max_patterns - 1, p_binomial)
        theoretical_pdf /= theoretical_pdf.sum()
        
        omega = np.round(theoretical_pdf * self.m_fake_users).astype(int)
        
        # Adjust for total
        diff = self.m_fake_users - np.sum(omega)
        if diff != 0:
            max_idx = np.argmax(omega)
            omega[max_idx] += diff
        
        return omega
    
    
    
    def generate_fake_reports(self) -> List[Union[np.ndarray, Dict[str, int], int]]:
        """
        Generate fake user reports following the adaptive pattern for the specified protocol
        """
        if self.protocol_type == 'OUE':
            return self._generate_fake_reports_oue()
        elif self.protocol_type == 'OLH':
            return self._generate_fake_reports_olh()
        else:
            raise ValueError(f"Fake report generation not implemented for {self.protocol_type}")
        
        
    def _generate_fake_reports_oue(self) -> List[np.ndarray]:
        fake_reports = []
        
        for k in range(len(self.omega)):
            num_fake_users_k = self.omega[k]
            
            for _ in range(num_fake_users_k):
                # Create a report with exactly k ones
                report = np.zeros(self.d, dtype=int)
                
                if k > 0:
                    # Always set target item to 1
                    report[self.target_item] = 1
                    remaining_ones = k - 1
                    
                    # Randomly select remaining positions for ones (excluding target)
                    if remaining_ones > 0:
                        available_positions = list(range(self.d))
                        available_positions.remove(self.target_item)
                        if remaining_ones <= len(available_positions):
                            selected_positions = np.random.choice(
                                available_positions, 
                                size=remaining_ones, 
                                replace=False
                            )
                            report[selected_positions] = 1
                        else:
                            # If more ones needed than available positions, fill all others
                            report[available_positions] = 1
                
                fake_reports.append(report)
        
        return fake_reports
    
    def _generate_fake_reports_olh(self) -> List[Dict[str, int]]:
        """Generate fake reports for OLH"""
        fake_reports = []
        
        for pattern_idx in range(len(self.omega)):
            num_fake_users = self.omega[pattern_idx]
            
            for _ in range(num_fake_users):
                # Generate a fake OLH report
                # Sample hash parameters
                a = np.random.randint(1, self.ldp_mechanism.prime)
                b = np.random.randint(0, self.ldp_mechanism.prime)
                
                # Hash the target item
                target_bucket = self.ldp_mechanism._hash(self.target_item, a, b)
                
                # With high probability, report the target bucket (biased toward target)
                if np.random.random() < 0.8:  # Bias toward target
                    y = target_bucket
                else:
                    # Sometimes report a different bucket to maintain plausibility
                    y = np.random.randint(0, self.ldp_mechanism.g)
                
                fake_reports.append({'a': a, 'b': b, 'y': y})
        
        return fake_reports
    
    
    def execute_attack(self, legitimate_data: Union[List[np.ndarray], List[Dict[str, int]], List[int]]) -> Union[List[np.ndarray], List[Dict[str, int]], List[int]]:
        """
        Execute the APA attack by injecting fake users
        
        Args:
            legitimate_data: List of legitimate reports
            
        Returns:
            Combined dataset with fake users injected
        """
        fake_reports = self.generate_fake_reports()
        
        print(f"\nGenerated {len(fake_reports)} fake reports for {self.protocol_type}")
        print(f"Legitimate users: {len(legitimate_data)}")
        
        # Combine legitimate and fake data
        attacked_data = legitimate_data + fake_reports
        
        print(f"Total users after attack: {len(attacked_data)}")
        
        return attacked_data