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
    
    # Process pairs of uniform random numbers
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

def analyze_chi_squared_distribution(chi_squared_values, df):
    print(f"\nChi-squared Distribution Analysis (df={df}):")
    print(f"Sample Size: {len(chi_squared_values)}")
    
    # Basic statistics
    mean = sum(chi_squared_values) / len(chi_squared_values)
    variance = sum((x - mean) ** 2 for x in chi_squared_values) / len(chi_squared_values)
    
    # Theoretical values
    theoretical_mean = df
    theoretical_var = 2 * df
    
    print(f"Mean: {mean:.4f} (Theoretical: {theoretical_mean:.4f})")
    print(f"Variance: {variance:.4f} (Theoretical: {theoretical_var:.4f})")
    
    # Quantile comparison
    chi_squared_values_sorted = sorted(chi_squared_values)
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    print("\nQuantile Comparison:")
    print(f"{'Quantile':<10}{'Generated':<15}{'Theoretical':<15}{'Difference':<15}")
    
    for q in quantiles:
        idx = int(q * len(chi_squared_values))
        sample_quantile = chi_squared_values_sorted[idx]
        theoretical_quantile = stats.chi2.ppf(q, df)
        diff = abs(sample_quantile - theoretical_quantile)
        
        print(f"{q:<10}{sample_quantile:<15.4f}{theoretical_quantile:<15.4f}{diff:<15.4f}")

def plot_chi_squared_distribution(chi_squared_values, df):
    plt.figure(figsize=(10, 6))
    
    # Filter out extreme values for better visualization
    max_display = np.percentile(chi_squared_values, 99)
    plot_values = [x for x in chi_squared_values if x <= max_display]
    
    # Plot histogram
    hist, bins, _ = plt.hist(plot_values, bins=50, density=True, alpha=0.6, 
                             label='Generated Chi-squared distribution')
    
    # Plot theoretical chi-squared PDF
    x = np.linspace(0.01, max_display, 1000)
    y = stats.chi2.pdf(x, df)
    plt.plot(x, y, 'r-', lw=2, label=f'Theoretical χ²({df}) PDF')
    
    # Plot settings
    plt.title(f'Chi-squared Distribution with {df} Degrees of Freedom')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # LCG parameters from lcg_parameters_calculator.py output
    a = 370  # Multiplier (from your calculator)
    c = 71   # Increment (from your calculator)
    m = 1107 # Modulus = 3^3 * 41
    seed = 1
    
    # Chi-squared distribution parameter
    df = 5  # Degrees of freedom
    
    # Sample size calculation
    # For each chi-squared value, we need df standard normals
    # Each Box-Muller transform needs 2 uniform random numbers to generate 2 standard normals
    target_sample_size = 1000  # Number of chi-squared values to generate
    needed_uniforms = math.ceil(target_sample_size * (df/2))
    
    # Generate uniform random numbers using your LCG
    print(f"Generating {needed_uniforms} uniform random numbers using LCG...")
    uniform_random_numbers = generate_uniform_random_numbers(a, c, m, seed, needed_uniforms)
    
    # Transform to standard normal variables using Box-Muller
    print("Transforming to standard normal distribution using Box-Muller transform...")
    standard_normals = box_muller_transform(uniform_random_numbers)
    
    # Generate chi-squared variables
    print(f"Generating chi-squared variables with {df} degrees of freedom...")
    chi_squared_values = generate_chi_squared(standard_normals, df)

    analyze_chi_squared_distribution(chi_squared_values, df)
    plot_chi_squared_distribution(chi_squared_values, df)

if __name__ == "__main__":
    main()
