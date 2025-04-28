import math
import matplotlib.pyplot as plt
import numpy as np
from math import gcd

def calculate_power(b, m):
    """Apskaičiuoja mažiausią s, kuriam b^s ≡ 0 mod m"""
    power = 1
    val = b % m
    while val != 0:
        val = (val * b) % m
        power += 1
    return power

def lcg_sequence(x0, a, c, m, length):
    """Sugeneruoja seką naudojant tiesinį kongruentinį generatorių."""
    sequence = [x0]
    x = x0
    
    for _ in range(length - 1):
        x = (a * x + c) % m
        sequence.append(x)
    
    return sequence

def calculate_correlation(sequence, m):
    """Apskaičiuoja lag-1 koreliaciją tarp gretimų sekos narių."""
    # Normalizuojame seką į [0,1]
    normalized = np.array(sequence) / m
    
    # Skaičiuojame lag-1 autokoreliaciją
    n = len(normalized)
    if n <= 1:
        return 0
    
    mean = np.mean(normalized)
    numerator = sum((normalized[i] - mean) * (normalized[i+1] - mean) for i in range(n-1))
    denominator = sum((x - mean) ** 2 for x in normalized)
    
    if denominator == 0:
        return 0
    
    return abs(numerator / denominator)  # Grąžiname absoliutų koreliacijos dydį

def find_best_c(a, m, x0=1, test_length=1000, candidates=50):
    """Testuoja skirtingas c reikšmes ir grąžina tą, kuri turi mažiausią koreliaciją."""
    c_candidates = []
    
    # Generuojame c kandidatus, kurie yra tarpusavyje pirminiai su m
    c = 1
    while len(c_candidates) < candidates and c < m:
        if gcd(c, m) == 1:
            c_candidates.append(c)
        c += 2  # Nagrinėjame tik nelyginius c
    
    best_c = 1
    min_correlation = float('inf')
    
    # Testuojame kiekvieną kandidatą
    for c in c_candidates:
        sequence = lcg_sequence(x0, a, c, m, test_length)
        correlation = calculate_correlation(sequence, m)
        
        if correlation < min_correlation:
            min_correlation = correlation
            best_c = c
    
    return best_c, min_correlation

def analyze_sequence(x0, a, c, m, length):
    """Analizuoja sugeneruotos sekos savybes."""
    sequence = lcg_sequence(x0, a, c, m, length)
    
    # Apskaičiuoja periodą
    seen = {}
    for i, x in enumerate(sequence):
        if x in seen:
            period = i - seen[x]
            break
        seen[x] = i
    else:
        period = "Nenustatytas"
    
    return sequence, period

def main():
    # 1. Nustatome pagrindinius parametrus
    m = 776  # Modulis m = 2^3 * 97
    b = 388  # b = LCM(4, 97)
    a = b + 1  # a = 389
    power = calculate_power(b, m)
    
    # Išvedame visus parametrus tvarkingai, kad galėtumėte palyginti rezultatus
    print("Parametrų nustatymai:")
    print("---------------------")
    print(f"Modulis m: {m}")
    print(f"Daugiklis a: {a}")
    print(f"b (a - 1): {b}")
    print(f"Galingumas s: {power}")
    
    # 2. Randame geriausią c reikšmę naudojant koreliacijų testus
    print("\nIeškoma geriausia c reikšmė...")
    x0 = 1  # pradinis sėklos skaičius
    best_c, correlation = find_best_c(a, m, x0)
    
    print(f"Geriausia c reikšmė: {best_c}")
    print(f"Gretimų narių koreliacija: {correlation:.6f}")
    
    # 3. Demonstruojame generatorių su rastais parametrais
    length = 20  # Demonstracijai parodome 20 narių
    print(f"\nSugeneruota seka naudojant a={a}, c={best_c}, x0={x0}:")
    sequence, period = analyze_sequence(x0, a, best_c, m, length)
    print(f"Pirmos {length} reikšmės: {sequence}")
    print(f"Nustatytas periodas: {period}")
    
    # 4. Vizualizuojame sugeneruotus skaičius
    # Sugeneruokime didesnę seką vizualizacijai
    long_sequence = lcg_sequence(x0, a, best_c, m, 1000)
    normalized = [x/m for x in long_sequence]

if __name__ == "__main__":
    main()