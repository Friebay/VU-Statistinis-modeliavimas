import math
import numpy as np
from lcg_parameters_calculator import generate_lcg_sequence

def monte_carlo_integration(func, a, b, num_samples=1000):
    """
    Perform Monte Carlo integration of a function over interval [a, b]
    
    Parameters:
        func (function): Function to integrate
        a (float): Lower bound of integration
        b (float): Upper bound of integration
        num_samples (int): Number of random points to use, default is 1000
        
    Returns:
        float: Estimated value of the integral
    """
    # Generate uniform random numbers using LCG
    # Parameters from lcg_parameters_calculator output
    lcg_a = 370  # Multiplier
    lcg_c = 71   # Increment
    lcg_m = 1107  # Modulus = 3^3 * 41
    seed = 1
    
    # Generate exactly the number of samples needed
    lcg_values = generate_lcg_sequence(lcg_a, lcg_c, lcg_m, seed, num_samples)
    
    # Convert to values in interval [a, b]
    uniform_samples = []
    for val in lcg_values:
        # Normalize to [0, 1] and then scale to [a, b]
        x = a + (b - a) * (val / lcg_m)
        uniform_samples.append(x)
    
    # Evaluate function at each sample point
    function_values = [func(x) for x in uniform_samples]
    
    # Calculate the average function value and multiply by interval width
    average_value = sum(function_values) / num_samples
    integral_estimate = (b - a) * average_value
    
    return integral_estimate, uniform_samples, function_values

def analytical_solution():
    """
    Calculate the analytical solution to the integral ∫[1 to π] (x(ln(x)+e^x))dx
    """
    # For ∫[1 to π] (x(ln(x)+e^x))dx
    # First term: ∫[1 to π] x ln(x) dx = [0.5x^2 ln(x) - 0.25x^2]_1^π
    # Second term: ∫[1 to π] x e^x dx requires integration by parts
    
    # First term evaluation at upper bound π
    term1_upper = 0.5 * (math.pi**2) * math.log(math.pi) - 0.25 * (math.pi**2)
    # First term evaluation at lower bound 1
    term1_lower = 0.5 * (1**2) * math.log(1) - 0.25 * (1**2) 
    # Note: ln(1) = 0, so term1_lower = -0.25
    
    # Second term: ∫[1 to π] x e^x dx = [x*e^x - e^x]_1^π
    term2_upper = math.pi * math.exp(math.pi) - math.exp(math.pi)
    term2_lower = 1 * math.exp(1) - math.exp(1)
    
    # Combine the terms
    result = (term1_upper - term1_lower) + (term2_upper - term2_lower)
    
    return result

def main():
    # Define the function to integrate: f(x) = x(ln(x) + e^x)
    def f(x):
        return x * (math.log(x) + math.exp(x))
    
    # Integration bounds
    a = 1
    b = math.pi
      # Calculate analytical solution
    exact_value = analytical_solution()
    print(f"Analytical solution: {exact_value:.10f}")
    print("\n1. Monte Carlo Integration with Uniform Distribution:")
    print(f"{'Sample Size':<15}{'Method':<20}{'Estimate':<20}{'Absolute Error':<20}{'Relative Error (%)':<20}")
    
    # Perform Monte Carlo integration with 1000 samples
    estimate, samples, function_values = monte_carlo_integration(f, a, b)
    abs_error = abs(estimate - exact_value)
    rel_error = abs_error / exact_value * 100
    print(f"{1000:<15}{'Uniform':<20}{estimate:<20.10f}{abs_error:<20.10f}{rel_error:<20.6f}")
    
    print("\nConclusion:")
    print(f"The integral ∫[1 to π] (x(ln(x)+e^x))dx ≈ {estimate:.10f} (Uniform sampling)")
    print(f"With uniform sampling, relative error: {rel_error:.6f}%")
    
if __name__ == "__main__":
    main()
