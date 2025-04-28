import numpy as np
from math import gcd

def prime_factorization(n):
    """Find prime factorization of a number n."""
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d*d > n and n > 1:
            factors.append(n)
            break
    return factors

def check_conditions(a, m):
    """Check if multiplier 'a' satisfies the conditions for maximum period."""
    b = a - 1
    
    # Get prime factorization of m
    prime_factors = prime_factorization(m)
    unique_factors = set(prime_factors)
    
    # Check if b is a multiple of all prime factors of m
    for factor in unique_factors:
        if b % factor != 0:
            return False
    
    # Check if 4 is a divisor of b when m is divisible by 4
    if m % 4 == 0 and b % 4 != 0:
        return False
    
    return True

def find_power(a, m):
    """Find the power of the sequence (smallest s where (a-1)^s ≡ 0 mod m)."""
    b = a - 1
    s = 1
    b_power = b
    
    while b_power % m != 0:
        b_power = (b_power * b) % m
        s += 1
        
        # Safeguard against infinite loops
        if s > 1000:
            return float('inf')
            
    return s

def lcg_sequence(x0, a, c, m, length):
    """Generate a sequence using Linear Congruential Generator."""
    sequence = [x0]
    x = x0
    
    for _ in range(length - 1):
        x = (a * x + c) % m
        sequence.append(x)
    
    return sequence

def calculate_correlation(sequence, m):
    """Calculate lag-1 correlation coefficient between adjacent terms in the sequence."""
    # Normalize the sequence to [0,1]
    normalized = np.array(sequence) / m
    
    # Calculate lag-1 autocorrelation
    n = len(normalized)
    if n <= 1:
        return 0
    
    mean = np.mean(normalized)
    numerator = sum((normalized[i] - mean) * (normalized[i+1] - mean) for i in range(n-1))
    denominator = sum((x - mean) ** 2 for x in normalized)
    
    if denominator == 0:
        return 0
    
    return abs(numerator / denominator)  # Return absolute value of correlation

def test_c_candidates(a, m, x0=1, test_length=1000, candidates=50):
    """Test different values of c and return the one with lowest correlation."""
    c_candidates = []
    
    # Generate candidate c values that are relatively prime to m
    c = 1
    while len(c_candidates) < candidates and c < m:
        if gcd(c, m) == 1:
            c_candidates.append(c)
        c += 2  # Only consider odd values for c
    
    best_c = 1
    min_correlation = float('inf')
    
    # Test each candidate
    for c in c_candidates:
        sequence = lcg_sequence(x0, a, c, m, test_length)
        correlation = calculate_correlation(sequence, m)
        
        if correlation < min_correlation:
            min_correlation = correlation
            best_c = c
    
    return best_c, min_correlation

def find_best_parameters(m, candidates_count=5):
    """Find the best parameters for LCG."""
    best_a_candidates = []
    
    # Try different values of 'a' and find those with maximum period
    for a in range(2, m):
        if gcd(a, m) == 1 and check_conditions(a, m):
            power = find_power(a, m)
            best_a_candidates.append((a, power))
            
            # Break after finding enough candidates
            if len(best_a_candidates) >= candidates_count:
                break
    
    # Sort by power (smaller is better)
    best_a_candidates.sort(key=lambda x: x[1])
    
    # Calculate appropriate c values for each a
    best_parameters = []
    for a, power in best_a_candidates:
        # Find best c value by testing correlation between adjacent terms
        c, correlation = test_c_candidates(a, m)
        
        best_parameters.append((a, c, power, correlation))
    
    return best_parameters

def analyze_sequence(x0, a, c, m, length):
    """Analyze properties of the generated sequence."""
    sequence = lcg_sequence(x0, a, c, m, length)
    
    # Calculate period
    seen = {}
    for i, x in enumerate(sequence):
        if x in seen:
            period = i - seen[x]
            break
        seen[x] = i
    else:
        period = "Not determined"
    
    return sequence, period

def main():
    # Set the modulus as per the requirement
    m = 776  # 2^3 * 97
    
    print(f"Modulis m = {m} = 2^3 * 97")
    
    # Find best parameters
    print("\nIeškomi geriausi parametrai...")
    best_parameters = find_best_parameters(m)
    
    print("\nGeriausi parametrai (a, c, galingumas, koreliacija):")
    for a, c, power, correlation in best_parameters:
        print(f"a = {a}, c = {c}, galingumas = {power}, koreliacija = {correlation:.6f}")
    
    # Optionally, demonstrate a sequence with the best parameters
    if best_parameters:
        a, c, _, _ = best_parameters[0]
        x0 = 1  # Starting seed
        length = 20  # Length of sequence to display
        
        print(f"\nSugeneruota seka naudojant a={a}, c={c}, x0={x0}:")
        sequence, period = analyze_sequence(x0, a, c, m, length)
        print(f"Pirmos {length} reikšmės: {sequence}")
        print(f"Nustatytas periodas: {period}")

if __name__ == "__main__":
    main()