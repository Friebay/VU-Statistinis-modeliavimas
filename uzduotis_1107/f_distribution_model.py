# failo kelias: c:\Users\zabit\Documents\GitHub\VU-Statistinis-modeliavimas\uzduotis_1107\f_distribution_model.py
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from lcg_parameters_calculator import generate_lcg_sequence

def generate_uniform_random_numbers(a, c, m, seed, length):
    sequence = generate_lcg_sequence(a, c, m, seed, length)
    # Normalizuoti į [0,1]
    return [x / m for x in sequence]

def box_muller_transform(uniform_random_numbers):
    standard_normals = []
    
    for i in range(0, len(uniform_random_numbers) - 1, 2):
        u1 = uniform_random_numbers[i]
        u2 = uniform_random_numbers[i + 1]
        
        # Išvengti log(0)
        if u1 == 0:
            u1 = 1e-10
            
        # Box-Muller transformacija
        r = math.sqrt(-2 * math.log(u1))
        theta = 2 * math.pi * u2
        
        z1 = r * math.cos(theta)
        z2 = r * math.sin(theta)
        
        standard_normals.extend([z1, z2])
    
    return standard_normals

def generate_chi_squared(standard_normals, df):
    chi_squared_values = []
    
    # Apdoroti grupėmis po df standartinių normaliųjų
    for i in range(0, len(standard_normals) - df + 1, df):
        group = standard_normals[i:i+df]
        
        # Kvadratų suma atitinka chi-kvadrato skirstinį su df laisvės laipsniais
        chi_squared = sum(z**2 for z in group)
        chi_squared_values.append(chi_squared)
    
    return chi_squared_values

def generate_f_distribution(chi_squared_v1, chi_squared_v2, v1, v2):
    """
    Generuoti F-pasiskirsčiusius atsitiktinius kintamuosius naudojant chi-kvadrato kintamuosius.
    
    Parametrai:
        chi_squared_v1 (list): Chi-kvadrato atsitiktiniai kintamieji su v1 laisvės laipsniais
        chi_squared_v2 (list): Chi-kvadrato atsitiktiniai kintamieji su v2 laisvės laipsniais
        v1 (int): Skaitiklio laisvės laipsniai
        v2 (int): Vardiklio laisvės laipsniai
    
    Grąžina:
        list: F-pasiskirsčiusius atsitiktinius kintamuosius
    """
    f_values = []
    
    # Naudoti formulę: X = (v2*Y1)/(v1*Y2)
    min_length = min(len(chi_squared_v1), len(chi_squared_v2))
    for i in range(min_length):
        # Išvengti dalybos iš nulio
        if chi_squared_v2[i] == 0:
            continue
            
        f_value = (v2 * chi_squared_v1[i]) / (v1 * chi_squared_v2[i])
        f_values.append(f_value)
    
    return f_values

def analyze_f_distribution(f_values, v1, v2):
    """
    Analizuoti sugeneruotą F-skirstinį.
    
    Parametrai:
        f_values (list): F-pasiskirsčiusių kintamųjų sąrašas
        v1 (int): Skaitiklio laisvės laipsniai
        v2 (int): Vardiklio laisvės laipsniai
    """
    print(f"\nF-skirstinio analizė (v1={v1}, v2={v2}):")
    print(f"Imties dydis: {len(f_values)}")
    
    # Pagrindinė statistika
    mean = sum(f_values) / len(f_values)
    variance = sum((x - mean) ** 2 for x in f_values) / len(f_values)
    
    # Teorinis vidurkis (egzistuoja tik kai v2 > 2)
    if v2 > 2:
        theoretical_mean = v2 / (v2 - 2)
        print(f"Vidurkis: {mean:.4f} (Teorinis: {theoretical_mean:.4f})")
    else:
        print(f"Vidurkis: {mean:.4f} (Teorinis vidurkis neegzistuoja kai v2 <= 2)")
    
    # Teorinė dispersija (egzistuoja tik kai v2 > 4)
    if v2 > 4:
        theoretical_var = (2 * v2**2 * (v1 + v2 - 2)) / (v1 * (v2 - 2)**2 * (v2 - 4))
        print(f"Dispersija: {variance:.4f} (Teorinė: {theoretical_var:.4f})")
    else:
        print(f"Dispersija: {variance:.4f} (Teorinė dispersija neegzistuoja kai v2 <= 4)")
    
    # Kvantilių palyginimas
    f_values_sorted = sorted(f_values)
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    print("\nKvantilių palyginimas:")
    print(f"{'Kvantilis':<10}{'Sugeneruota':<15}{'Teorinis':<15}{'Skirtumas':<15}")
    
    for q in quantiles:
        idx = int(q * len(f_values))
        sample_quantile = f_values_sorted[idx]
        theoretical_quantile = stats.f.ppf(q, v1, v2)
        diff = abs(sample_quantile - theoretical_quantile)
        
        print(f"{q:<10}{sample_quantile:<15.4f}{theoretical_quantile:<15.4f}{diff:<15.4f}")

def plot_f_distribution(f_values, v1, v2):
    plt.figure(figsize=(10, 6))
    
    # Filtruoti ekstremalias reikšmes geresnei vizualizacijai
    max_display = np.percentile(f_values, 99)
    plot_values = [x for x in f_values if x <= max_display]
    
    # Vaizduoti histogramą
    hist, bins, _ = plt.hist(plot_values, bins=100, density=True, alpha=0.6, 
                             label='Sugeneruotas F-skirstinys')
    
    # Vaizduoti teorinę F-skirstinio tankio funkciją
    x = np.linspace(0.01, max_display, 1000)
    y = stats.f.pdf(x, v1, v2)
    plt.plot(x, y, 'r-', lw=2, label=f'Teorinė F({v1},{v2}) tankio funkcija')
    
    # Diagramos nustatymai
    plt.title(f'F-skirstinys su {v1} ir {v2} laisvės laipsniais')
    plt.xlabel('Reikšmė')
    plt.ylabel('Tikimybės tankis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Išsaugoti diagramą
    plt.tight_layout()
    plt.show()

def main():
    a = 370
    c = 71
    m = 1107
    seed = 1
    
    v1 = 2  # Skaitiklio laisvės laipsniai
    v2 = 3  # Vardiklio laisvės laipsniai
    
    # Imties dydžio skaičiavimas
    # Kiekvienai F-pasiskirsčiusiai reikšmei reikia:
    # - v1 standartinių normaliųjų chi-kvadratui su v1 laisvės laipsniais
    # - v2 standartinių normaliųjų chi-kvadratui su v2 laisvės laipsniais
    # Kiekviena Box-Muller transformacija reikalauja 2 tolygiai pasiskirsčiusių atsitiktinių skaičių 2 standartiniams normaliesiems sugeneruoti
    
    target_sample_size = 1000
    needed_uniforms = math.ceil(target_sample_size * (v1 + v2) / 2)
    
    # Generuoti tolygiai pasiskirsčiusius atsitiktinius skaičius naudojant LCG
    print(f"Generuojami {needed_uniforms} tolygiai pasiskirstyti atsitiktiniai skaičiai naudojant LCG...")
    uniform_random_numbers = generate_uniform_random_numbers(a, c, m, seed, needed_uniforms)
    
    # Transformuoti į standartinį normalųjį skirstinį naudojant Box-Muller
    print("Transformuojama į standartinį normalųjį skirstinį naudojant Box-Muller transformaciją...")
    standard_normals = box_muller_transform(uniform_random_numbers)
    
    # Generuoti chi-kvadrato kintamuosius
    print(f"Generuojami chi-kvadrato kintamieji su {v1} ir {v2} laisvės laipsniais...")
    chi_squared_v1 = generate_chi_squared(standard_normals, v1)
    chi_squared_v2 = generate_chi_squared(standard_normals, v2)
    
    # Generuoti F-pasiskirsčiusius kintamuosius naudojant formulę X = (v2*Y1)/(v1*Y2)
    print("Skaičiuojami F-pasiskirsčiusių atsitiktiniai kintamieji...")
    f_values = generate_f_distribution(chi_squared_v1, chi_squared_v2, v1, v2)
    
    analyze_f_distribution(f_values, v1, v2)
    
    plot_f_distribution(f_values, v1, v2)

if __name__ == "__main__":
    main()
