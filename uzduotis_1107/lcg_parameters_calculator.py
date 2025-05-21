import math
from math import gcd
import numpy as np
import scipy.stats as stats


def calculate_theoretical_correlation(a, c, m):
    # C ≈ (1/a) * (1 - 6(c/m) + 6(c/m)²)
    c_m_ratio = c / m
    correlation = (1/a) * (1 - 6 * c_m_ratio + 6 * (c_m_ratio ** 2))
    
    return abs(correlation)

def serial_test_triplets(sequence):
    # Generuoti nepersidengiančius trejetus iš sekos pagal užduoties aprašymą
    # Imame grupėmis po 3 elementus: (Y_0, Y_1, Y_2), (Y_3, Y_4, Y_5), ...
    usable_length = (len(sequence) // 3) * 3
    triplets = [tuple(sequence[i:i+3]) for i in range(0, usable_length, 3)]
    
    # Apibrėžti visus galimus trejetus
    possible_triplets = [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]
    # Skaičiuoti stebėtų kiekvieno trejeto dažnius
    observed_counts = {triplet: 0 for triplet in possible_triplets}
    for triplet in triplets:
        observed_counts[triplet] += 1
    # Apskaičiuoti tikėtiną dažnį
    total_triplets = len(triplets)
    expected_count = total_triplets / 8  # Vienoda tikimybė kiekvienam trejeto (1/8)
    
    # Apskaičiuoti Chi-kvadrato statistiką
    chi_squared = sum((observed_counts[triplet] - expected_count) ** 2 / expected_count
                      for triplet in possible_triplets)
    
    # Laisvės laipsniai
    degrees_of_freedom = len(possible_triplets) - 1  # 8 - 1 = 7
    # Apskaičiuoti p-reikšmę
    p_value = 1 - stats.chi2.cdf(chi_squared, degrees_of_freedom)

    return {
        "chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "observed_counts": observed_counts,
        "expected_count": expected_count
    }
def monotonicity_test(sequence):

    increasing_runs = {1: 0, 2: 0, 3: 0, '>3': 0}
    decreasing_runs = {1: 0, 2: 0, 3: 0, '>3': 0}
    
    current_run_type = None  # Didėjantis arba mažėjantis
    run_length = 1

    for i in range(1, len(sequence)):
        if sequence[i] > sequence[i-1]:
            # Current pair is increasing
            if current_run_type == "increasing":
                # Continue the current increasing run
                run_length += 1
            else:
                # End previous run if it exists
                if current_run_type == "decreasing":
                    # Record the decreasing run that just ended
                    # Adjust run length to represent number of comparisons (elements - 1)
                    adjusted_length = run_length - 1
                    if adjusted_length == 1:
                        decreasing_runs[1] += 1
                    elif adjusted_length == 2:
                        decreasing_runs[2] += 1
                    elif adjusted_length == 3:
                        decreasing_runs[3] += 1
                    else:  # adjusted_length > 3
                        decreasing_runs['>3'] += 1
                
                # Start a new increasing run
                current_run_type = "increasing"
                run_length = 2  # Current element + previous element
        
        elif sequence[i] < sequence[i-1]:
            # Current pair is decreasing
            if current_run_type == "decreasing":
                # Continue the current decreasing run
                run_length += 1
            else:
                # End previous run if it exists
                if current_run_type == "increasing":
                    # Record the increasing run that just ended
                    # Adjust run length to represent number of comparisons (elements - 1)
                    adjusted_length = run_length - 1
                    if adjusted_length == 1:
                        increasing_runs[1] += 1
                    elif adjusted_length == 2:
                        increasing_runs[2] += 1
                    elif adjusted_length == 3:
                        increasing_runs[3] += 1
                    else:  # adjusted_length > 3
                        increasing_runs['>3'] += 1
                
                # Start a new decreasing run
                current_run_type = "decreasing"
                run_length = 2  # Current element + previous element
        
        else:  # Jeigu lygūs
            # Equal values - handle as a plateau
            # For the purpose of this test, we'll end any current run
            if current_run_type == "increasing":
                # Record the increasing run
                # Adjust run length to represent number of comparisons (elements - 1)
                adjusted_length = run_length - 1
                if adjusted_length == 1:
                    increasing_runs[1] += 1
                elif adjusted_length == 2:
                    increasing_runs[2] += 1
                elif adjusted_length == 3:
                    increasing_runs[3] += 1
                else:  # adjusted_length > 3
                    increasing_runs['>3'] += 1
                
                # Reset run tracking
                current_run_type = None
                run_length = 1
                
            elif current_run_type == "decreasing":
                # Record the decreasing run
                # Adjust run length to represent number of comparisons (elements - 1)
                adjusted_length = run_length - 1
                if adjusted_length == 1:
                    decreasing_runs[1] += 1
                elif adjusted_length == 2:
                    decreasing_runs[2] += 1
                elif adjusted_length == 3:
                    decreasing_runs[3] += 1
                else:  # adjusted_length > 3
                    decreasing_runs['>3'] += 1
                
                # Reset run tracking
                current_run_type = None
                run_length = 1
    
    # Don't forget to count the last run
    if current_run_type == "increasing":
        # Adjust run length to represent number of comparisons (elements - 1)
        adjusted_length = run_length - 1
        if adjusted_length == 1:
            increasing_runs[1] += 1
        elif adjusted_length == 2:
            increasing_runs[2] += 1
        elif adjusted_length == 3:
            increasing_runs[3] += 1
        else:  # adjusted_length > 3
            increasing_runs['>3'] += 1
            
    elif current_run_type == "decreasing":
        # Adjust run length to represent number of comparisons (elements - 1)
        adjusted_length = run_length - 1
        if adjusted_length == 1:
            decreasing_runs[1] += 1
        elif adjusted_length == 2:
            decreasing_runs[2] += 1
        elif adjusted_length == 3:
            decreasing_runs[3] += 1
        else:  # adjusted_length > 3
            decreasing_runs['>3'] += 1
    
    total_runs = {}
    for length in [1, 2, 3, '>3']:
        total_runs[length] = increasing_runs[length] + decreasing_runs[length]
    
    # Teorinės tikimybės
    n = len(sequence)
    expected_all = (2*n + 1) / 3
    expected_1 = (5*n + 6) / 12 
    expected_2 = (11*n - 3) / 60
    
    k = 3
    expected_3 = (2*((k**2 + 3*k + 1)*n - (k**3 + 2*k**2 - 4*k - 5))) / math.factorial(k + 3)
    
    expected_gt3 = expected_all - expected_1 - expected_2 - expected_3
    
    expected_runs = {
        1: expected_1,
        2: expected_2,
        3: expected_3,
        '>3': expected_gt3
    }

    chi_squared = sum(((total_runs[length] - expected_runs[length])**2) / expected_runs[length] 
                     for length in [1, 2, 3, '>3'])
    
    # p reikšmė su 3 laisvės laipsniais, nes  4 grupės (1, 2, 3 ir >3) minus 1
    p_value = 1 - stats.chi2.cdf(chi_squared, df=3)
    
    return {
        "increasing_runs": increasing_runs,
        "decreasing_runs": decreasing_runs,
        "total_runs": total_runs,
        "expected_runs": expected_runs,
        "chi_squared": chi_squared,
        "p_value": p_value,
    }

def prime_factorization(n):
    # Skaičiaus n pirminių daugiklių radimas
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
    """Rasti sekos galią (mažiausią s, kur (a-1)^s ≡ 0 mod m)."""
    b = a - 1
    s = 1
    result = b % m
    
    # Maksimalus iteracijų skaičius, kad būtų išvengta begalinio ciklo
    max_iterations = m
    iterations = 0
    
    while result != 0 and iterations < max_iterations:
        result = (result * b) % m
        s += 1
        iterations += 1
    
    # Jei neradome galios, kuri tenkina b^s ≡ 0 (mod m)
    if result != 0:
        return 0
    
    return s

def find_valid_c(m):
    valid_c_values = []
    
    for c in range(1, m):
        if gcd(c, m) == 1:
            # Jei m dalijasi iš 4, c turi būti nelyginis
            if m % 4 == 0 and c % 2 == 0:
                continue
            valid_c_values.append(c)
    
    return valid_c_values

def generate_lcg_sequence(a, c, m, seed, length):
    """Generuoti atsitiktinių skaičių seką naudojant LCG algoritmą."""
    sequence = [seed]
    x = seed
    
    for _ in range(length - 1):
        x = (a * x + c) % m
        sequence.append(x)
    
    return sequence

def test_c_correlation(a, m, valid_c_values, num_tests):
    """Calculate theoretical correlation for different c values."""
    c_theoretical_correlations = []
    
    for c in valid_c_values[:num_tests]:
        theoretical_correlation = calculate_theoretical_correlation(a, c, m)
        c_theoretical_correlations.append((c, theoretical_correlation))
    
    # Sort by theoretical correlation (lower is better)
    c_theoretical_correlations.sort(key=lambda x: x[1])
    
    return c_theoretical_correlations

def main():
    m = 1107

    valid_a_values = []
    
    print(f"Modulio m = {m} pirminiai daugikliai: {prime_factorization(m)}")
    
    for a in range(2, m):
        # Tikrinti tik reikšmes, kur gcd(a,m) = 1
        if gcd(a, m) == 1:
            b = a - 1
            # Patikrinti, ar b tenkina mūsų sąlygas
            # Kai m = 1107 = 3^3 * 41:
            # b turėtų dalintis iš 3 ir 41
            if b % 3 == 0 and b % 41 == 0:
                power = find_power(a, m)
                valid_a_values.append((a, b, power))
    # Rūšiuoti pagal galią (didesnė yra geresnė)
    valid_a_values.sort(key=lambda x: x[2], reverse=True)
    
    # Spausdinti rezultatus
    print(f"\n{'Daugiklis a':<15}{'b=a-1':<15}{'Galingumas s':<15}")
    
    for a, b, power in valid_a_values[:10]:  # Rodyti 10 geriausių rezultatų
        print(f"{a:<15}{b:<15}{power:<15}")
    
    # Geriausias rezultatas
    best_a, best_b, best_power = valid_a_values[0]
    print("\nGeriausias rezultatas:")
    print(f"Daugiklis a = {best_a}")
    print(f"b = a - 1 = {best_b}")
    print(f"Galingumas s = {best_power}")
    
    # Patikrinti rezultatą
    print("\nPatikrinimas:")
    print(f"b^s mod m = {best_b}^{best_power} mod {m} = {pow(best_b, best_power, m)}")
    # Rasti tinkamas c reikšmes
    valid_c_values = find_valid_c(m)
    
    theoretical_correlations = test_c_correlation(best_a, m, valid_c_values, num_tests=1000)

    print("\n=== Teorinės koreliacijos rezultatai ===")
    print(f"{'c reikšmė':<15}{'Teorinė koreliacija':<22}")
    for c, corr in theoretical_correlations[:3]:  # Rodyti 3 geriausius rezultatus
        print(f"{c:<15}{corr:.6f}")
    
    # Geriausia c reikšmė pagal teorinę formulę
    best_c_theo, best_corr_theo = theoretical_correlations[0]
    
    print("\nUžduočiai naudosime (pagal teorinę formulę):")
    print(f"c = {best_c_theo} (teorinė koreliacija: {best_corr_theo:.6f})")
    
    best_c = best_c_theo
    
    print("\nPilni tiesinio kongruentinio metodo parametrai:")
    print(f"a = {best_a}")
    print(f"c = {best_c}")
    print(f"m = {m}")
    print(f"Tiesinio kongruentinio metodo formulė: X_n+1 = ({best_a} * X_n + {best_c}) mod {m}")
    seed = 1
    sequence = generate_lcg_sequence(best_a, best_c, m, seed, length=100)

    print("\nPirmieji 100 sugeneruotų pseudoatsitiktinių skaičių:")
    print(sequence)

    long_sequence = generate_lcg_sequence(best_a, best_c, m, seed, length=1000)

    # Sugeneruoti dvejetainę seką iš LCG sekos (pvz., modulo 2)
    binary_sequence = [x % 2 for x in long_sequence]

    # Skaičiuoti perėjimus sekoje
    transitions = []
    for i in range(len(binary_sequence)-1):
        transitions.append((binary_sequence[i], binary_sequence[i+1]))
    
    # Spausdinti perėjimų skaičių
    transition_counts = {}
    for t in transitions[:100]:
        if t in transition_counts:
            transition_counts[t] += 1
        else:
            transition_counts[t] = 1
    
    print("\nPerėjimų skaičius (pirmi 100 perėjimų):")
    for t, count in transition_counts.items():
        print(f"{t}: {count}")    # Atlikti nuoseklumo testą trejetams
    serial_test_results = serial_test_triplets(binary_sequence)
    
    # Spausdinti nuoseklumo testo rezultatus
    print("\nNuoseklumo testo trejetams rezultatai:")
    print(f"Laisvės laipsniai: {serial_test_results['degrees_of_freedom']}")
    print(f"P-reikšmė: {serial_test_results['p_value']:.4f}")     
      # Run the monotonicity test
    monotonicity_test_results = monotonicity_test(long_sequence)

    print("\nMonotoniškumo testo rezultatai:")
    print(f"{'Run Length':<12}{'Observed':<12}{'Expected':<12}{'Difference':<12}")
    print("-" * 48)
    for length in [1, 2, 3, '>3']:
        observed = monotonicity_test_results['total_runs'][length]
        expected = monotonicity_test_results['expected_runs'][length]
        diff = observed - expected
        print(f"{length:<12}{observed:<12.2f}{expected:<12.2f}{diff:<12.2f}")
    print("-" * 48)

    # Sum row
    total_observed = sum(monotonicity_test_results['total_runs'].values())
    total_expected = sum(monotonicity_test_results['expected_runs'].values())
    print(f"{'Total':<12}{total_observed:<12.2f}{total_expected:<12.2f}{total_observed-total_expected:<12.2f}")

    print(f"\nChi-squared statistic: {monotonicity_test_results['chi_squared']:.4f}")
    print(f"P-value: {monotonicity_test_results['p_value']:.4f}")

if __name__ == "__main__":
    main()