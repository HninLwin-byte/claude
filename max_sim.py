"""
Altitude and Heading Similarity Optimization
Maximizes similarity between ATC commands and ADS-B actual data
with physical and operational constraints
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple
import json


class AltitudeHeadingOptimizer:
    """
    Optimize matching between expected (ATC) and current (ADS-B) altitude and heading
    Objective: Maximize similarity score
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize optimizer with constraint parameters
        
        Args:
            config: Dictionary with constraint parameters
        """
        self.config = config or self.get_default_config()
        
    @staticmethod
    def get_default_config() -> Dict:
        """Default configuration with aviation constraints"""
        return {
            # Altitude constraints
            'max_climb_rate': 3000,      # feet/minute
            'max_descent_rate': 3000,    # feet/minute
            'altitude_tolerance': 200,    # feet (±200 for en-route)
            'min_altitude': 0,           # feet
            'max_altitude': 50000,       # feet
            
            # Heading constraints
            'max_turn_rate': 3,          # degrees/second
            'heading_tolerance': 10,     # degrees (±10)
            'min_heading': 0,            # degrees
            'max_heading': 360,          # degrees
            
            # Time constraints
            'time_to_comply': 120,       # seconds (2 minutes)
            'adsb_update_rate': 1,       # seconds
            
            # Measurement uncertainty
            'altitude_uncertainty': 25,   # feet (GPS accuracy)
            'heading_uncertainty': 1,     # degrees
            
            # Weights for multi-objective
            'altitude_weight': 0.6,
            'heading_weight': 0.4
        }
    
    def calculate_altitude_similarity(self, expected_alt: float, current_alt: float) -> float:
        """
        Calculate similarity score for altitude
        Uses Gaussian similarity: higher when closer, decreases with distance
        
        Args:
            expected_alt: Commanded altitude (feet)
            current_alt: Actual ADS-B altitude (feet)
            
        Returns:
            Similarity score [0, 1]
        """
        diff = abs(expected_alt - current_alt)
        tolerance = self.config['altitude_tolerance']
        
        # Gaussian similarity: exp(-diff²/2σ²)
        # σ = tolerance, so similarity = 0.607 at tolerance boundary
        sigma = tolerance
        similarity = np.exp(-(diff ** 2) / (2 * sigma ** 2))
        
        return float(similarity)
    
    def calculate_heading_similarity(self, expected_hdg: float, current_hdg: float) -> float:
        """
        Calculate similarity score for heading (handles 0/360 wrap-around)
        
        Args:
            expected_hdg: Commanded heading (degrees)
            current_hdg: Actual ADS-B heading (degrees)
            
        Returns:
            Similarity score [0, 1]
        """
        # Handle circular nature of heading (0° = 360°)
        diff = abs(expected_hdg - current_hdg)
        if diff > 180:
            diff = 360 - diff
        
        tolerance = self.config['heading_tolerance']
        
        # Gaussian similarity
        sigma = tolerance
        similarity = np.exp(-(diff ** 2) / (2 * sigma ** 2))
        
        return float(similarity)
    
    def calculate_combined_similarity(self, expected_alt: float, current_alt: float,
                                     expected_hdg: float, current_hdg: float) -> Dict:
        """
        Calculate weighted combined similarity score
        
        Returns:
            Dictionary with individual and combined scores
        """
        alt_sim = self.calculate_altitude_similarity(expected_alt, current_alt)
        hdg_sim = self.calculate_heading_similarity(expected_hdg, current_hdg)
        
        w_alt = self.config['altitude_weight']
        w_hdg = self.config['heading_weight']
        
        combined_sim = w_alt * alt_sim + w_hdg * hdg_sim
        
        return {
            'altitude_similarity': alt_sim,
            'heading_similarity': hdg_sim,
            'combined_similarity': combined_sim,
            'altitude_diff': abs(expected_alt - current_alt),
            'heading_diff': min(abs(expected_hdg - current_hdg), 
                               360 - abs(expected_hdg - current_hdg))
        }
    
    def check_altitude_constraint(self, alt: float) -> bool:
        """Check if altitude satisfies constraints"""
        return (self.config['min_altitude'] <= alt <= self.config['max_altitude'])
    
    def check_heading_constraint(self, hdg: float) -> bool:
        """Check if heading satisfies constraints"""
        return (0 <= hdg <= 360)
    
    def check_altitude_rate_constraint(self, alt_start: float, alt_end: float, 
                                       time_delta: float) -> bool:
        """
        Check if altitude change rate is within physical limits
        
        Args:
            alt_start: Starting altitude (feet)
            alt_end: Ending altitude (feet)
            time_delta: Time elapsed (seconds)
            
        Returns:
            True if rate is feasible
        """
        if time_delta == 0:
            return alt_start == alt_end
        
        rate = (alt_end - alt_start) / (time_delta / 60)  # feet/minute
        
        if rate > 0:  # Climbing
            return rate <= self.config['max_climb_rate']
        else:  # Descending
            return abs(rate) <= self.config['max_descent_rate']
    
    def check_heading_rate_constraint(self, hdg_start: float, hdg_end: float,
                                      time_delta: float) -> bool:
        """
        Check if heading change rate is within physical limits
        
        Args:
            hdg_start: Starting heading (degrees)
            hdg_end: Ending heading (degrees)
            time_delta: Time elapsed (seconds)
            
        Returns:
            True if turn rate is feasible
        """
        if time_delta == 0:
            return hdg_start == hdg_end
        
        # Handle wrap-around
        diff = abs(hdg_end - hdg_start)
        if diff > 180:
            diff = 360 - diff
        
        turn_rate = diff / time_delta  # degrees/second
        
        return turn_rate <= self.config['max_turn_rate']
    
    def objective_function(self, x: np.ndarray, expected_alt: float, expected_hdg: float) -> float:
        """
        Objective function for optimization (to be MINIMIZED)
        We want to MAXIMIZE similarity, so we return negative similarity
        
        Args:
            x: [altitude, heading] to optimize
            expected_alt: Target altitude
            expected_hdg: Target heading
            
        Returns:
            Negative similarity (for minimization)
        """
        current_alt, current_hdg = x
        
        result = self.calculate_combined_similarity(expected_alt, current_alt,
                                                    expected_hdg, current_hdg)
        
        # Return negative because we minimize but want to maximize similarity
        return -result['combined_similarity']
    
    def optimize_trajectory(self, initial_state: Dict, expected_state: Dict,
                           time_delta: float = 60) -> Dict:
        """
        Optimize trajectory from initial to expected state
        Finds best achievable altitude and heading given constraints
        
        Args:
            initial_state: {'altitude': float, 'heading': float}
            expected_state: {'altitude': float, 'heading': float}
            time_delta: Time to achieve target (seconds)
            
        Returns:
            Optimized state and similarity metrics
        """
        initial_alt = initial_state['altitude']
        initial_hdg = initial_state['heading']
        expected_alt = expected_state['altitude']
        expected_hdg = expected_state['heading']
        
        # Calculate maximum achievable altitude change
        max_alt_change = self.config['max_climb_rate'] * (time_delta / 60)
        alt_bounds = (
            max(self.config['min_altitude'], initial_alt - max_alt_change),
            min(self.config['max_altitude'], initial_alt + max_alt_change)
        )
        
        # Calculate maximum achievable heading change
        max_hdg_change = self.config['max_turn_rate'] * time_delta
        # For simplicity, allow full range (handle wrap-around in constraint)
        hdg_bounds = (0, 360)
        
        # Define constraints
        constraints = []
        
        # Altitude rate constraint
        def altitude_rate_constraint(x):
            alt = x[0]
            rate = (alt - initial_alt) / (time_delta / 60)
            if rate > 0:
                return self.config['max_climb_rate'] - rate
            else:
                return self.config['max_descent_rate'] - abs(rate)
        
        constraints.append({'type': 'ineq', 'fun': altitude_rate_constraint})
        
        # Heading rate constraint
        def heading_rate_constraint(x):
            hdg = x[1]
            diff = abs(hdg - initial_hdg)
            if diff > 180:
                diff = 360 - diff
            turn_rate = diff / time_delta
            return self.config['max_turn_rate'] - turn_rate
        
        constraints.append({'type': 'ineq', 'fun': heading_rate_constraint})
        
        # Initial guess: move towards expected state
        x0 = np.array([
            np.clip(expected_alt, alt_bounds[0], alt_bounds[1]),
            expected_hdg
        ])
        
        # Bounds
        bounds = [alt_bounds, hdg_bounds]
        
        # Optimize using scipy
        result = minimize(
            fun=self.objective_function,
            x0=x0,
            args=(expected_alt, expected_hdg),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimized_alt, optimized_hdg = result.x
        
        # Calculate final similarity
        similarity_metrics = self.calculate_combined_similarity(
            expected_alt, optimized_alt, expected_hdg, optimized_hdg
        )
        
        return {
            'initial_state': initial_state,
            'expected_state': expected_state,
            'optimized_state': {
                'altitude': float(optimized_alt),
                'heading': float(optimized_hdg)
            },
            'similarity_metrics': similarity_metrics,
            'is_compliant': similarity_metrics['combined_similarity'] >= 0.8,
            'optimization_success': result.success,
            'time_delta': time_delta
        }
    
    def assess_compliance(self, test_cases: List[Dict]) -> List[Dict]:
        """
        Assess compliance for multiple test cases
        
        Args:
            test_cases: List of dicts with 'expected' and 'current' states
            
        Returns:
            List of results with similarity scores
        """
        results = []
        
        for idx, case in enumerate(test_cases):
            expected_alt = case['expected']['altitude']
            expected_hdg = case['expected']['heading']
            current_alt = case['current']['altitude']
            current_hdg = case['current']['heading']
            
            # Calculate similarity
            similarity_metrics = self.calculate_combined_similarity(
                expected_alt, current_alt, expected_hdg, current_hdg
            )
            
            # Check constraints
            constraints_satisfied = {
                'altitude_valid': self.check_altitude_constraint(current_alt),
                'heading_valid': self.check_heading_constraint(current_hdg),
                'altitude_rate_valid': self.check_altitude_rate_constraint(
                    case.get('previous_altitude', current_alt),
                    current_alt,
                    case.get('time_delta', 1)
                ),
                'heading_rate_valid': self.check_heading_rate_constraint(
                    case.get('previous_heading', current_hdg),
                    current_hdg,
                    case.get('time_delta', 1)
                )
            }
            
            result = {
                'case_id': idx,
                'expected': case['expected'],
                'current': case['current'],
                'similarity_metrics': similarity_metrics,
                'constraints': constraints_satisfied,
                'is_compliant': (
                    similarity_metrics['combined_similarity'] >= 0.8 and
                    all(constraints_satisfied.values())
                )
            }
            
            results.append(result)
        
        return results


def generate_sample_data() -> List[Dict]:
    """Generate sample test data (not real data as specified)"""
    return [
        {
            'expected': {'altitude': 10000, 'heading': 90},
            'current': {'altitude': 10050, 'heading': 92},
            'previous_altitude': 9500,
            'previous_heading': 85,
            'time_delta': 30
        },
        {
            'expected': {'altitude': 15000, 'heading': 180},
            'current': {'altitude': 15200, 'heading': 175},
            'previous_altitude': 14500,
            'previous_heading': 170,
            'time_delta': 45
        },
        {
            'expected': {'altitude': 8000, 'heading': 270},
            'current': {'altitude': 8500, 'heading': 280},
            'previous_altitude': 9000,
            'previous_heading': 275,
            'time_delta': 60
        },
        {
            'expected': {'altitude': 12000, 'heading': 45},
            'current': {'altitude': 11800, 'heading': 43},
            'previous_altitude': 11500,
            'previous_heading': 40,
            'time_delta': 30
        },
        {
            'expected': {'altitude': 20000, 'heading': 360},
            'current': {'altitude': 19500, 'heading': 355},
            'previous_altitude': 18000,
            'previous_heading': 350,
            'time_delta': 90
        }
    ]


def main():
    """Main execution"""
    print("="*80)
    print("ALTITUDE & HEADING SIMILARITY OPTIMIZATION")
    print("="*80)
    
    # Initialize optimizer
    optimizer = AltitudeHeadingOptimizer()
    
    print("\nConstraints Configuration:")
    print(json.dumps(optimizer.config, indent=2))
    
    # Generate sample data
    test_cases = generate_sample_data()
    
    print(f"\n\nAnalyzing {len(test_cases)} test cases...")
    print("="*80)
    
    # Assess compliance
    results = optimizer.assess_compliance(test_cases)
    
    # Print results
    for result in results:
        print(f"\nCase {result['case_id'] + 1}:")
        print(f"  Expected: ALT={result['expected']['altitude']}ft, HDG={result['expected']['heading']}°")
        print(f"  Current:  ALT={result['current']['altitude']}ft, HDG={result['current']['heading']}°")
        print(f"  \nSimilarity Scores:")
        print(f"    Altitude:  {result['similarity_metrics']['altitude_similarity']:.4f}")
        print(f"    Heading:   {result['similarity_metrics']['heading_similarity']:.4f}")
        print(f"    Combined:  {result['similarity_metrics']['combined_similarity']:.4f}")
        print(f"  \nDifferences:")
        print(f"    Altitude:  {result['similarity_metrics']['altitude_diff']:.1f} feet")
        print(f"    Heading:   {result['similarity_metrics']['heading_diff']:.1f}°")
        print(f"  \nConstraints: {result['constraints']}")
        print(f"  Compliant: {'✓' if result['is_compliant'] else '✗'}")
        print("-"*80)
    
    # Summary statistics
    compliant_count = sum(1 for r in results if r['is_compliant'])
    avg_similarity = np.mean([r['similarity_metrics']['combined_similarity'] for r in results])
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total cases: {len(results)}")
    print(f"Compliant: {compliant_count} ({compliant_count/len(results)*100:.1f}%)")
    print(f"Average similarity: {avg_similarity:.4f}")
    
    # Example: Optimization
    print(f"\n{'='*80}")
    print("TRAJECTORY OPTIMIZATION EXAMPLE")
    print(f"{'='*80}")
    
    initial_state = {'altitude': 8000, 'heading': 90}
    expected_state = {'altitude': 12000, 'heading': 180}
    
    opt_result = optimizer.optimize_trajectory(initial_state, expected_state, time_delta=120)
    
    print(f"\nInitial:  ALT={initial_state['altitude']}ft, HDG={initial_state['heading']}°")
    print(f"Expected: ALT={expected_state['altitude']}ft, HDG={expected_state['heading']}°")
    print(f"Optimized: ALT={opt_result['optimized_state']['altitude']:.0f}ft, "
          f"HDG={opt_result['optimized_state']['heading']:.0f}°")
    print(f"\nOptimized Similarity: {opt_result['similarity_metrics']['combined_similarity']:.4f}")
    print(f"Compliant: {'✓' if opt_result['is_compliant'] else '✗'}")
    

if __name__ == "__main__":
    main()
