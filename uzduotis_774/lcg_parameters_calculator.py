import math
from math import gcd
import numpy as np
import scipy.stats as stats

def frequency_test(sequence, alpha=0.05):
    """
    Perform the Frequency (Monobit) Test on a binary sequence.
    
    Parameters:
        sequence (list): A list of integers (0 or 1) representing the binary sequence.
        alpha (float): Significance level for the test (default is 0.05).
    
    Returns:
        dict: A dictionary containing the test results.
    """
    n = len(sequence)
    if n == 0:
        raise ValueError("Sequence cannot be empty")
    
    # Convert 0/1 to -1/1 and compute sum
    s = sum(2*bit - 1 for bit in sequence)
    
    # Compute p-value (using complementary error function)
    p_value = math.erfc((abs(s) / math.sqrt(n)) / math.sqrt(2))
    
    # Conclusion
    conclusion = "Fail to reject null hypothesis" if p_value > alpha else "Reject null hypothesis"
    
    return {
        "statistic": abs(s) / math.sqrt(n),
        "p_value": p_value,
        "conclusion": conclusion
    }

def serial_test_triplets(sequence, alpha=0.05):
    """
    Perform the Serial Test for non-overlapping triplets (m=3) on a binary sequence (modulo 2).
    
    This implementation follows the book's description, using independent groups
    of three elements (Y_3k, Y_3k+1, Y_3k+2) rather than overlapping triplets.
    
    Parameters:
        sequence (list): A list of integers (0 or 1) representing the binary sequence.
        alpha (float): Significance level for the test (default is 0.05).
    
    Returns:
        dict: A dictionary containing the test results:
            - chi_squared: The Chi-squared statistic.
            - degrees_of_freedom: Degrees of freedom for the test.
            - p_value: The p-value of the test.
            - conclusion: Test conclusion based on the significance level.
            - observed_counts: Observed frequencies of triplets.
            - expected_count: Expected frequency for each triplet.
            - reliability: Assessment of test reliability based on expected counts.
    """
    # Validate input
    if not all(x in [0, 1] for x in sequence):
        raise ValueError("Sequence must contain only 0s and 1s.")
    if len(sequence) < 3:
        raise ValueError("Sequence length must be at least 3.")
    
    # Generate non-overlapping triplets from the sequence as per the book's description
    # We take groups of 3 elements: (Y_0, Y_1, Y_2), (Y_3, Y_4, Y_5), ...
    usable_length = (len(sequence) // 3) * 3
    triplets = [tuple(sequence[i:i+3]) for i in range(0, usable_length, 3)]
    
    # Define all possible triplets
    possible_triplets = [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]
    
    # Count observed frequencies of each triplet
    observed_counts = {triplet: 0 for triplet in possible_triplets}
    for triplet in triplets:
        observed_counts[triplet] += 1
      # Calculate expected frequency
    total_triplets = len(triplets)
    expected_count = total_triplets / 8  # Equal probability for each triplet (1/8)
    
    # Check if expected count is sufficient for reliable Chi-squared test
    reliability = "Very good reliable results" if expected_count > 20 else \
                 "Satisfactory results" if expected_count > 5 else \
                 "Unreliable results (expected count < 5)"
    
    # Calculate Chi-squared statistic
    chi_squared = sum((observed_counts[triplet] - expected_count) ** 2 / expected_count
                      for triplet in possible_triplets)
    
    # Degrees of freedom
    degrees_of_freedom = len(possible_triplets) - 1  # 8 - 1 = 7
    
    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi_squared, degrees_of_freedom)
    
    # Conclusion
    conclusion = "Fail to reject null hypothesis" if p_value > alpha else "Reject null hypothesis"

    return {
        "chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "conclusion": conclusion,
        "observed_counts": observed_counts,
        "expected_count": expected_count,
        "reliability": reliability
    }

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

def main():
    m = 776
    print(f"Modulis m = {m} = 2^3 * 97")
    print("\nIeškome daugiklio 'a' su maksimaliu periodu ir galingumu:")
    
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
        c_correlations = test_c_correlation(best_a, m, valid_c_values)
        
        print(f"\n{'c reikšmė':<15}{'Koreliacija':<15}")
        for c, corr in c_correlations[:5]:  # Show top 10 results
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
        
        seed = 1
        sequence = generate_lcg_sequence(best_a, best_c, m, seed, length=5)
        
        print("\nPirmieji 5 sugeneruotų pseudoatsitiktinių skaičių:")
        print(f"{'n':<5}{'X_n':<8}{'X_n/m':<10}")
        for i, number in enumerate(sequence):
            normalized = number / m
            print(f"{i:<5}{number:<8}{normalized:.6f}")

        long_sequence = generate_lcg_sequence(best_a, best_c, m, seed, length=1000)
        
        print(long_sequence[:10])

        # Generate a binary sequence from the LCG sequence (e.g., modulo 2)
        binary_sequence = [x % 2 for x in long_sequence]

        # Print the first 20 values of the binary sequence to check the pattern
        print("\nFirst 20 values of binary sequence (modulo 2):")
        binary_str = ''.join(str(bit) for bit in binary_sequence[:20])
        print(binary_str)
        
        # Count transitions in the sequence
        transitions = []
        for i in range(len(binary_sequence)-1):
            transitions.append((binary_sequence[i], binary_sequence[i+1]))
        
        # Print the transitions count
        transition_counts = {}
        for t in transitions[:100]:  # Count first 100 transitions
            if t in transition_counts:
                transition_counts[t] += 1
            else:
                transition_counts[t] = 1
        
        print("\nTransition counts (first 100 transitions):")
        for t, count in transition_counts.items():
            print(f"{t}: {count}")

        # Perform the Serial Test for triplets
        alpha = 0.05  # Significance level
        serial_test_results = serial_test_triplets(binary_sequence, alpha)
        
        # Print the results of the Serial Test
        print("\nSerial Test for Triplets Results:")
        print(f"Chi-squared Statistic: {serial_test_results['chi_squared']:.4f}")
        print(f"Degrees of Freedom: {serial_test_results['degrees_of_freedom']}")
        print(f"P-value: {serial_test_results['p_value']:.4f}")
        print(f"Conclusion: {serial_test_results['conclusion']}")
        print(f"Reliability: {serial_test_results['reliability']}")
        print("\nObserved Counts:")
        for triplet, count in serial_test_results['observed_counts'].items():
            print(f"{triplet}: {count}")
        print(f"Expected Count per Triplet: {serial_test_results['expected_count']:.4f}")

        # Perform the Frequency (Monobit) Test
        frequency_test_results = frequency_test(binary_sequence, alpha)
        
        # Print the results of the Frequency Test
        print("\nFrequency (Monobit) Test Results:")
        print(f"Test Statistic: {frequency_test_results['statistic']:.4f}")
        print(f"P-value: {frequency_test_results['p_value']:.4f}")
        print(f"Conclusion: {frequency_test_results['conclusion']}")

    else:
        print("Nerasta tinkamų daugiklio 'a' reikšmių.")

if __name__ == "__main__":
    main()