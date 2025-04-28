import math
from math import gcd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def serial_test_triplets(sequence, alpha=0.05):
    """
    Performs the Serial Test for triplets (m=3) on a binary sequence.
    
    Parameters:
    sequence (list or array-like): A sequence of integers (0s and 1s)
    alpha (float): Significance level for hypothesis testing, default 0.05
    
    Returns:
    dict: A dictionary containing the test results
    """
    import numpy as np
    from scipy import stats
    
    # Input validation
    if not isinstance(sequence, (list, np.ndarray)):
        raise TypeError("Sequence must be a list or numpy array")
    
    if len(sequence) < 3:
        raise ValueError("Sequence must have at least 3 elements for triplet analysis")
    
    # Convert to numpy array for easier handling
    seq = np.array(sequence)
    
    # Check that sequence contains only 0s and 1s
    if not np.all(np.isin(seq, [0, 1])):
        raise ValueError("Sequence must contain only 0s and 1s")
    
    # Get sequence length
    n = len(seq)
    
    # Generate all possible triplets (000, 001, ..., 111)
    all_triplets = [(i, j, k) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
    
    # Count occurrences of each triplet
    observed_counts = {}
    for triplet in all_triplets:
        observed_counts[triplet] = 0
    
    # Count overlapping triplets in the sequence
    for i in range(n - 2):
        triplet = tuple(seq[i:i+3])
        observed_counts[triplet] += 1
    
    # Calculate expected count for each triplet
    # For a random sequence, each triplet should appear with equal probability
    expected_count = (n - 2) / 8  # Number of triplets divided by number of possible triplets (2^3 = 8)
    
    # Calculate Chi-squared statistic
    chi_squared = sum((observed - expected_count)**2 / expected_count 
                      for observed in observed_counts.values())
    
    # Degrees of freedom = Number of categories - 1
    degrees_of_freedom = 7  # 8 triplets - 1
    
    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi_squared, degrees_of_freedom)
    
    # Determine if null hypothesis should be rejected
    reject_null = p_value < alpha
    
    # Format conclusion
    if reject_null:
        conclusion = f"Reject null hypothesis (p-value = {p_value:.4f} < {alpha}). " \
                     f"The sequence likely exhibits non-randomness in triplet patterns."
    else:
        conclusion = f"Fail to reject null hypothesis (p-value = {p_value:.4f} >= {alpha}). " \
                     f"No significant evidence of non-randomness in triplet patterns."
    
    # Prepare results
    results = {
        "chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "conclusion": conclusion,
        "observed_counts": observed_counts,
        "expected_count": expected_count,
        "alpha": alpha
    }
    
    return results

def plot_correlation(sequence, m, a, c, valid_c_values):
    """Plot correlation visualizations for the LCG sequence."""
    normalized = [x/m for x in sequence]
    
    plt.figure(figsize=(15, 6))
    
    # Scatter plot of consecutive values
    plt.subplot(1, 2, 1)
    plt.scatter(normalized[:-1], normalized[1:], alpha=0.5, s=10)
    plt.title(f'Scatter Plot of X_n vs X_n+1 (a={a}, c={c}, m={m})')
    plt.xlabel('X_n / m')
    plt.ylabel('X_n+1 / m')
    plt.grid(True, alpha=0.3)
    
    # Correlation bar chart for different c values
    plt.subplot(1, 2, 2)
    c_subset = valid_c_values[:100]  # First 100 valid c values
    correlations = [lag1_pearson(generate_lcg_sequence(a, c_val, m, 1), m) for c_val in c_subset]
    
    plt.bar(range(len(c_subset)), correlations)
    plt.title('Correlation for Different c Values')
    plt.xlabel('c Value Index')
    plt.ylabel('Absolute Correlation')
    plt.xticks(range(len(c_subset)), c_subset, rotation=90)
    
    plt.tight_layout()
    plt.savefig('lcg_correlation.png')
    plt.show()


def prime_factorization(n):
    """Find prime factorization of a number n."""
    factors = {}
    d = 2
    while d*d <= n:
        while n % d == 0:
            if d in factors:
                factors[d] += 1
            else:
                factors[d] = 1
            n //= d
        d += 1
    if n > 1:
        if n in factors:
            factors[n] += 1
        else:
            factors[n] = 1
    return factors

def find_power(a, m):
    """Find the power of the sequence (smallest s where (a-1)^s ≡ 0 mod m)."""
    b = a - 1
    s = 1
    result = b % m
    
    # Maximum iterations to prevent infinite loop
    max_iterations = m
    iterations = 0
    
    while result != 0 and iterations < max_iterations:
        result = (result * b) % m
        s += 1
        iterations += 1
    
    # If we didn't find a power that makes b^s ≡ 0 (mod m)
    if result != 0:
        return 0  # Return 0 instead of None to avoid formatting errors
    
    return s

def find_valid_c(m):
    """Find a valid increment 'c' for the LCG.
    
    For maximum period:
    - c must be relatively prime to m (gcd(c,m) = 1)
    - If m is a multiple of 4, c should be odd
    """
    valid_c_values = []
    
    for c in range(1, m):
        if gcd(c, m) == 1:
            # If m is divisible by 4, c should be odd
            if m % 4 == 0 and c % 2 == 0:
                continue
            valid_c_values.append(c)
    
    return valid_c_values

def generate_lcg_sequence(a, c, m, seed, length=1000):
    """Generate a sequence of random numbers using LCG."""
    sequence = [seed]
    x = seed
    
    for _ in range(length - 1):
        x = (a * x + c) % m
        sequence.append(x)
    
    return sequence

def calculate_correlation(sequence):
    """Calculate correlation between adjacent terms in the sequence."""
    # Normalize sequence to [0,1] range for proper statistical analysis
    normalized = [x / (len(sequence) - 1) for x in sequence]
    
    # Calculate correlation between consecutive terms
    x = normalized[:-1]  # All but the last element
    y = normalized[1:]   # All but the first element
    
    # Calculate Pearson correlation coefficient
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
    denominator_x = sum((val - x_mean) ** 2 for val in x)
    denominator_y = sum((val - y_mean) ** 2 for val in y)
    
    # Avoid division by zero
    if denominator_x == 0 or denominator_y == 0:
        return 1.0  # Return high correlation if division by zero would occur
    
    correlation = numerator / (math.sqrt(denominator_x) * math.sqrt(denominator_y))
    
    # Return absolute value as we want minimum correlation regardless of direction
    return abs(correlation)

def test_c_correlation(a, m, valid_c_values, num_tests=50, seed=1):
    """Test different c values and find the one with minimal adjacent term correlation."""
    c_correlations = []
    
    for c in valid_c_values[:num_tests]:  # Test a subset of values to save time
        sequence = generate_lcg_sequence(a, c, m, seed)
        correlation = calculate_correlation(sequence)
        c_correlations.append((c, correlation))
    
    # Sort by correlation (lower is better)
    c_correlations.sort(key=lambda x: x[1])
    
    return c_correlations

import numpy as np

def lag1_pearson(sequence, m):
    """
    Pearsono koreliacija tarp X[n] ir X[n+1],
    normalizuota į [0,1] dalinant iš m.
    """
    arr = np.array(sequence) / m
    x, y = arr[:-1], arr[1:]
    return abs(np.corrcoef(x, y)[0, 1])

def main():
    # Given modulus
    m = 776  # 2^3 * 97
    
    print(f"Modulis m = {m} = 2^3 * 97")
    print("\nIeškome daugiklio 'a' su maksimaliu periodu ir galimgumu:")
    
    # Find all valid 'a' values
    valid_a_values = []
    
    # Print prime factorization for debugging
    print(f"Modulio m pirminiai daugikliai: {prime_factorization(m)}")
    
    for a in range(2, m):
        # Only check values where gcd(a,m) = 1
        if gcd(a, m) == 1:
            b = a - 1
            # Check if b satisfies our conditions
            # For m = 776 = 2^3 * 97:
            # b should be divisible by both 2 and 97
            # Also, since m is divisible by 8 (2^3), b should be divisible by 4
            if b % 2 == 0 and b % 97 == 0 and b % 4 == 0:
                power = find_power(a, m)
                valid_a_values.append((a, b, power))
                print(f"Patikrinta a={a}, b={b}, galingumas={power}")
    
    # Sort by power (larger is better)
    valid_a_values.sort(key=lambda x: x[2], reverse=True)
    
    # Print results
    if valid_a_values:
        print(f"\n{'Daugiklis a':<15}{'b=a-1':<15}{'Galingumas s':<15}")
        
        for a, b, power in valid_a_values[:10]:  # Show top 10 results
            print(f"{a:<15}{b:<15}{power:<15}")
        
        # Best result
        best_a, best_b, best_power = valid_a_values[0]
        print("\nGeriausias rezultatas:")
        print(f"Daugiklis a = {best_a}")
        print(f"b = a - 1 = {best_b}")
        print(f"Power s = {best_power}")
        
        # Verify the result
        print("\nPatikrinimas:")
        print(f"b^s mod m = {best_b}^{best_power} mod {m} = {pow(best_b, best_power, m)}")
        
        # Find valid c values
        valid_c_values = find_valid_c(m)
        
        # Test c values for correlation
        print("\nTiriame koreliacijas tarp gretimų narių skirtingoms c reikšmėms:")
        c_correlations = test_c_correlation(best_a, m, valid_c_values)
        
        print(f"\n{'c reikšmė':<15}{'Koreliacija':<15}")
        for c, corr in c_correlations[:10]:  # Show top 10 results
            print(f"{c:<15}{corr:.6f}")
        
        # Best c value (with minimal correlation)
        best_c, best_corr = c_correlations[0]
        
        print("\nMonte Carlo užduočiai rekomenduojama c reikšmė:")
        print(f"c = {best_c} (koreliacija: {best_corr:.6f})")
        
        print("\nPilni LCG parametrai:")
        print(f"a = {best_a}")
        print(f"c = {best_c}")
        print(f"m = {m}")
        print(f"LCG formulė: X_n+1 = ({best_a} * X_n + {best_c}) mod {m}")
        
        # Generate and display the first 10 pseudorandom numbers
        seed = 1  # Initial seed value
        sequence = generate_lcg_sequence(best_a, best_c, m, seed, length=10)
        
        print("\nPirmieji 10 sugeneruotų pseudoatsitiktinių skaičių:")
        print(f"{'n':<5}{'X_n':<8}{'X_n/m':<10}")
        for i, number in enumerate(sequence):
            normalized = number / m
            print(f"{i:<5}{number:<8}{normalized:.6f}")

        # Generate a longer sequence for visualization
        print("\nGeneruojamos vizualizacijos...")
        long_sequence = generate_lcg_sequence(best_a, best_c, m, seed, length=1000)
        
        # Convert LCG sequence to binary (using least significant bit)
        binary_sequence = [x % 2 for x in long_sequence]
        
        # Run the serial test
        serial_test_results = serial_test_triplets(binary_sequence)
        
        # Print the results
        print("\nSerial Test for Randomness (Triplets):")
        print(f"Chi-squared statistic: {serial_test_results['chi_squared']:.4f}")
        print(f"Degrees of freedom: {serial_test_results['degrees_of_freedom']}")
        print(f"p-value: {serial_test_results['p_value']:.4f}")
        print(f"\nConclusion: {serial_test_results['conclusion']}")
        
        # Create visualizations
        # plot_correlation(long_sequence, m, best_a, best_c, valid_c_values)

    else:
        print("Nerasta tinkamų daugiklio 'a' reikšmių.")

if __name__ == "__main__":
    main()