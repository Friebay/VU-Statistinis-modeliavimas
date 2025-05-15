import math
from math import gcd
import numpy as np
import scipy.stats as stats

def frequency_test(sequence, alpha=0.05):
    n = len(sequence)
    
    # Konvertuoti 0/1 į -1/1 ir apskaičiuoti sumą
    s = sum(2*bit - 1 for bit in sequence)
    
    # Apskaičiuoti p-reikšmę (naudojant papildomą klaidos funkciją)
    p_value = math.erfc((abs(s) / math.sqrt(n)) / math.sqrt(2))
    
    # Išvada
    conclusion = "Nepavyko atmesti nulinės hipotezės" if p_value > alpha else "Atmesti nulinę hipotezę"
    
    return {
        "statistic": abs(s) / math.sqrt(n),
        "p_value": p_value,
        "conclusion": conclusion
    }

def serial_test_triplets(sequence, alpha=0.05):
    # Patikrinti įvestį
    if not all(x in [0, 1] for x in sequence):
        raise ValueError("Seka turi susidėti tik iš 0 ir 1.")
    if len(sequence) < 3:
        raise ValueError("Sekos ilgis turi būti bent 3.")
    
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
    
    # Išvada
    conclusion = "Nepavyko atmesti nulinės hipotezės" if p_value > alpha else "Atmetame nulinę hipotezę"

    return {
        "chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "conclusion": conclusion,
        "observed_counts": observed_counts,
        "expected_count": expected_count
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
    """Rasti tinkamą prieaugį 'c' LCG algoritmui.
    
    Maksimaliam periodui:
    - c turi būti tarpusavyje pirminiai su m (gcd(c,m) = 1)
    - Jei m dalijasi iš 4, c turėtų būti nelyginis
    """
    valid_c_values = []
    
    for c in range(1, m):
        if gcd(c, m) == 1:
            # Jei m dalijasi iš 4, c turi būti nelyginis
            if m % 4 == 0 and c % 2 == 0:
                continue
            valid_c_values.append(c)
    
    return valid_c_values

def generate_lcg_sequence(a, c, m, seed, length=1000):
    """Generuoti atsitiktinių skaičių seką naudojant LCG algoritmą."""
    sequence = [seed]
    x = seed
    
    for _ in range(length - 1):
        x = (a * x + c) % m
        sequence.append(x)
    
    return sequence

def calculate_correlation(sequence):
    """Apskaičiuoti koreliaciją tarp gretimų sekos narių."""
    # Normalizuoti seką į [0,1] diapazoną tinkamam statistiniam analizui
    normalized = [x / (len(sequence) - 1) for x in sequence]
    
    # Apskaičiuoti koreliaciją tarp nuoseklių narių
    x = normalized[:-1]  # Visi, išskyrus paskutinį elementą
    y = normalized[1:]   # Visi, išskyrus pirmą elementą
    
    # Apskaičiuoti Pearson koreliacijos koeficientą
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
    denominator_x = sum((val - x_mean) ** 2 for val in x)
    denominator_y = sum((val - y_mean) ** 2 for val in y)
    
    # Išvengti dalybos iš nulio
    if denominator_x == 0 or denominator_y == 0:
        return 1.0  # Grąžinti aukštą koreliaciją, jei įvyktų dalyba iš nulio
    
    correlation = numerator / (math.sqrt(denominator_x) * math.sqrt(denominator_y))
    
    # Grąžinti absoliučią reikšmę, nes norime minimalios koreliacijos nepriklausomai nuo krypties
    return abs(correlation)

def test_c_correlation(a, m, valid_c_values, num_tests=50, seed=1):
    """Išbandyti skirtingas c reikšmes ir rasti tą, kuri turi minimalią gretimų narių koreliaciją."""
    c_correlations = []
    
    for c in valid_c_values[:num_tests]:  # Išbandyti poaibį reikšmių, kad sutaupytume laiko
        sequence = generate_lcg_sequence(a, c, m, seed)
        correlation = calculate_correlation(sequence)
        c_correlations.append((c, correlation))
    
    # Rūšiuoti pagal koreliaciją (mažesnė yra geresnė)
    c_correlations.sort(key=lambda x: x[1])
    
    return c_correlations

def main():
    m = 1107
    print(f"Modulis m = {m} = 3^3 * 41")
    print("\nIeškome daugiklio 'a' su maksimaliu periodu ir galingumu:")
    
    valid_a_values = []
    
    print(f"Modulio m pirminiai daugikliai: {prime_factorization(m)}")
    
    for a in range(2, m):
        # Tikrinti tik reikšmes, kur gcd(a,m) = 1
        if gcd(a, m) == 1:
            b = a - 1
            # Patikrinti, ar b tenkina mūsų sąlygas
            # Kai m = 1107 = 3^3 * 41:
            # b turėtų dalintis iš 3 ir 41
            # Taip pat, kadangi m dalijasi iš 27 (3^3), b turėtų dalintis iš 9
            if b % 3 == 0 and b % 41 == 0 and b % 9 == 0:
                power = find_power(a, m)
                valid_a_values.append((a, b, power))
                print(f"Patikrinta a={a}, b={b}, galingumas={power}")    # Rūšiuoti pagal galią (didesnė yra geresnė)
    valid_a_values.sort(key=lambda x: x[2], reverse=True)
    
    # Spausdinti rezultatus
    if valid_a_values:
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
        print(f"b^s mod m = {best_b}^{best_power} mod {m} = {pow(best_b, best_power, m)}")        # Rasti tinkamas c reikšmes
        valid_c_values = find_valid_c(m)
        
        # Testuoti c reikšmes dėl koreliacijos
        c_correlations = test_c_correlation(best_a, m, valid_c_values)
        
        print(f"\n{'c reikšmė':<15}{'Koreliacija':<15}")
        for c, corr in c_correlations[:5]:  # Rodyti 5 geriausius rezultatus
            print(f"{c:<15}{corr:.6f}")
        
        # Geriausia c reikšmė (su minimalia koreliacija)
        best_c, best_corr = c_correlations[0]
        
        print("\nUžduočiai naudosime:")
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

        # Sugeneruoti dvejetainę seką iš LCG sekos (pvz., modulo 2)
        binary_sequence = [x % 2 for x in long_sequence]

        # Spausdinti pirmas 20 dvejetainės sekos reikšmių patikrinti šabloną
        print("\nPirmos 20 dvejetainės sekos reikšmių (modulis 2):")
        binary_str = ''.join(str(bit) for bit in binary_sequence[:20])
        print(binary_str)
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
            print(f"{t}: {count}")        # Atlikti nuoseklumo testą tripletams
        alpha = 0.05  # Reikšmingumo lygis
        serial_test_results = serial_test_triplets(binary_sequence, alpha)
        
        # Spausdinti nuoseklumo testo rezultatus
        print("\nNuoseklumo testo tripletams rezultatai:")
        print(f"Chi-kvadrato statistika: {serial_test_results['chi_squared']:.4f}")
        print(f"Laisvės laipsniai: {serial_test_results['degrees_of_freedom']}")
        print(f"P-reikšmė: {serial_test_results['p_value']:.4f}")
        print(f"Išvada: {serial_test_results['conclusion']}")
        print("\nStebėti skaičiai:")
        for triplet, count in serial_test_results['observed_counts'].items():
            print(f"{triplet}: {count}")
        print(f"Tikėtinas kiekvieno tripleto skaičius: {serial_test_results['expected_count']:.4f}")        # Atlikti dažnio (Monobit) testą
        frequency_test_results = frequency_test(binary_sequence, alpha)
        
        # Spausdinti dažnio testo rezultatus
        print("\nDažnio (Monobit) testo rezultatai:")
        print(f"Testo statistika: {frequency_test_results['statistic']:.4f}")
        print(f"P-reikšmė: {frequency_test_results['p_value']:.4f}")
        print(f"Išvada: {frequency_test_results['conclusion']}")

    else:
        print("Nerasta tinkamų daugiklio 'a' reikšmių.")

if __name__ == "__main__":
    main()