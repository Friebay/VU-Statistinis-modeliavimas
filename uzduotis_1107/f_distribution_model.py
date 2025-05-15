import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from lcg_parameters_calculator import generate_lcg_sequence

def generate_uniform_random_numbers(a, c, m, seed, length):
    sequence = generate_lcg_sequence(a, c, m, seed, length)
    # Normalize to [0,1]
    return [x / m for x in sequence]

def box_muller_transform(uniform_random_numbers):
    standard_normals = []
    
    for i in range(0, len(uniform_random_numbers) - 1, 2):
        u1 = uniform_random_numbers[i]
        u2 = uniform_random_numbers[i + 1]
        
        # Avoid log(0)
        if u1 == 0:
            u1 = 1e-10
            
        # Box-Muller transformation
        r = math.sqrt(-2 * math.log(u1))
        theta = 2 * math.pi * u2
        
        z1 = r * math.cos(theta)
        z2 = r * math.sin(theta)
        
        standard_normals.extend([z1, z2])
    
    return standard_normals

def generate_chi_squared(standard_normals, df):
    chi_squared_values = []
    
    # Process in groups of df standard normals
    for i in range(0, len(standard_normals) - df + 1, df):
        group = standard_normals[i:i+df]
        
        # Sum of squares follows chi-squared distribution with df degrees of freedom
        chi_squared = sum(z**2 for z in group)
        chi_squared_values.append(chi_squared)
    
    return chi_squared_values

def generate_f_distribution(chi_squared_v1, chi_squared_v2, v1, v2):
    """
    Generate F-distributed random variables using chi-squared variables.
    
    Parameters:
        chi_squared_v1 (list): Chi-squared random variables with v1 degrees of freedom
        chi_squared_v2 (list): Chi-squared random variables with v2 degrees of freedom
        v1 (int): Numerator degrees of freedom
        v2 (int): Denominator degrees of freedom
    
    Returns:
        list: F-distributed random variables
    """
    f_values = []
    
    # Use formula: X = (v2*Y1)/(v1*Y2)
    min_length = min(len(chi_squared_v1), len(chi_squared_v2))
    for i in range(min_length):
        # Avoid division by zero
        if chi_squared_v2[i] == 0:
            continue
            
        f_value = (v2 * chi_squared_v1[i]) / (v1 * chi_squared_v2[i])
        f_values.append(f_value)
    
    return f_values

def analyze_f_distribution(f_values, v1, v2):
    """
    Analyze the generated F-distribution.
    
    Parameters:
        f_values (list): F-distributed random variables
        v1 (int): Numerator degrees of freedom
        v2 (int): Denominator degrees of freedom
    """
    print(f"\nF-Distribution Analysis (v1={v1}, v2={v2}):")
    print(f"Sample Size: {len(f_values)}")
    
    # Basic statistics
    mean = sum(f_values) / len(f_values)
    variance = sum((x - mean) ** 2 for x in f_values) / len(f_values)
    
    # Theoretical mean (exists only for v2 > 2)
    if v2 > 2:
        theoretical_mean = v2 / (v2 - 2)
        print(f"Mean: {mean:.4f} (Theoretical: {theoretical_mean:.4f})")
    else:
        print(f"Mean: {mean:.4f} (Theoretical mean doesn't exist for v2 <= 2)")
    
    # Theoretical variance (exists only for v2 > 4)
    if v2 > 4:
        theoretical_var = (2 * v2**2 * (v1 + v2 - 2)) / (v1 * (v2 - 2)**2 * (v2 - 4))
        print(f"Variance: {variance:.4f} (Theoretical: {theoretical_var:.4f})")
    else:
        print(f"Variance: {variance:.4f} (Theoretical variance doesn't exist for v2 <= 4)")
    
    # Quantile comparison
    f_values_sorted = sorted(f_values)
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    print("\nQuantile Comparison:")
    print(f"{'Quantile':<10}{'Generated':<15}{'Theoretical':<15}{'Difference':<15}")
    
    for q in quantiles:
        idx = int(q * len(f_values))
        sample_quantile = f_values_sorted[idx]
        theoretical_quantile = stats.f.ppf(q, v1, v2)
        diff = abs(sample_quantile - theoretical_quantile)
        
        print(f"{q:<10}{sample_quantile:<15.4f}{theoretical_quantile:<15.4f}{diff:<15.4f}")

def plot_f_distribution(f_values, v1, v2):
    plt.figure(figsize=(10, 6))
    
    # Filter out extreme values for better visualization
    max_display = np.percentile(f_values, 99)
    plot_values = [x for x in f_values if x <= max_display]
    
    # Plot histogram
    hist, bins, _ = plt.hist(plot_values, bins=100, density=True, alpha=0.6, 
                             label='Generated F-distribution')
    
    # Plot theoretical F-distribution PDF
    x = np.linspace(0.01, max_display, 1000)
    y = stats.f.pdf(x, v1, v2)
    plt.plot(x, y, 'r-', lw=2, label=f'Theoretical F({v1},{v2}) PDF')
    
    # Plot settings
    plt.title(f'F-Distribution with {v1} and {v2} Degrees of Freedom')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.show()

def main():
    a = 370
    c = 71
    m = 1107
    seed = 1
    
    v1 = 2  # Numerator degrees of freedom
    v2 = 3  # Denominator degrees of freedom
    
    # Sample size calculation
    # For each F-distributed value, we need:
    # - v1 standard normals for chi-squared with v1 df
    # - v2 standard normals for chi-squared with v2 df
    # Each Box-Muller transform needs 2 uniform random numbers to generate 2 standard normals
    
    target_sample_size = 1000
    needed_uniforms = math.ceil(target_sample_size * (v1 + v2) / 2)
    
    # Generate uniform random numbers using your LCG
    print(f"Generating {needed_uniforms} uniform random numbers using LCG...")
    uniform_random_numbers = generate_uniform_random_numbers(a, c, m, seed, needed_uniforms)
    
    # Transform to standard normal variables using Box-Muller
    print("Transforming to standard normal distribution using Box-Muller transform...")
    standard_normals = box_muller_transform(uniform_random_numbers)
    
    # Generate chi-squared variables
    print(f"Generating chi-squared variables with {v1} and {v2} degrees of freedom...")
    chi_squared_v1 = generate_chi_squared(standard_normals, v1)
    chi_squared_v2 = generate_chi_squared(standard_normals, v2)
    
    # Generate F-distributed variables using the formula X = (v2*Y1)/(v1*Y2)
    print("Computing F-distributed random variables...")
    f_values = generate_f_distribution(chi_squared_v1, chi_squared_v2, v1, v2)
    
    analyze_f_distribution(f_values, v1, v2)
    
    plot_f_distribution(f_values, v1, v2)

if __name__ == "__main__":
    main()
