import math
from math import gcd
import numpy as np

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

def test_c_correlation(a, m, valid_c_values, num_tests=50, seed=42):
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
        
        print("\nTinkami pridėtinio nario c variantai:")
        print(f"Rasta {len(valid_c_values)} tinkamų reikšmių.")
        print("Pirmosios 10 reikšmių:", valid_c_values[:10])
        
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
        
        # Original method
        recommended_c = valid_c_values[0] if valid_c_values else None
        
        print("\nPilni LCG parametrai:")
        print(f"a = {best_a}")
        print(f"c = {best_c}  (pasirinktas mažiausiai koreliuotiems gretimų narių)")
        print(f"m = {m}")
        print(f"LCG formulė: X_n+1 = ({best_a} * X_n + {best_c}) mod {m}")
    else:
        print("Nerasta tinkamų daugiklio 'a' reikšmių.")

if __name__ == "__main__":
    main()