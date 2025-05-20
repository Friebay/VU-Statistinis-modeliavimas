# failo kelias: c:\Users\zabit\Documents\GitHub\VU-Statistinis-modeliavimas\uzduotis_1107\chi_squared_distribution_model.py
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
    
    # Apdoroti tolygiai pasiskirsčiusių atsitiktinių skaičių poras
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
    
    # Apdoroti standartinį normalųjį skirstinį grupėmis po df
    for i in range(0, len(standard_normals) - df + 1, df):
        group = standard_normals[i:i+df]
        
        # Kvadratų suma atitinka chi kvadrato skirstinį su df laisvės laipsniais
        chi_squared = sum(z**2 for z in group)
        chi_squared_values.append(chi_squared)
    
    return chi_squared_values

def analyze_chi_squared_distribution(chi_squared_values, df):
    print(f"\nChi-kvadrato skirstinio analizė (df={df}):")
    print(f"Imties dydis: {len(chi_squared_values)}")
    
    # Pagrindinė statistika
    mean = sum(chi_squared_values) / len(chi_squared_values)
    variance = sum((x - mean) ** 2 for x in chi_squared_values) / len(chi_squared_values)
    
    # Teorinės vertės
    theoretical_mean = df
    theoretical_var = 2 * df
    
    print(f"Vidurkis: {mean:.4f} (Teorinis: {theoretical_mean:.4f})")
    print(f"Dispersija: {variance:.4f} (Teorinė: {theoretical_var:.4f})")
    
    # Kvantilių palyginimas
    chi_squared_values_sorted = sorted(chi_squared_values)
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    print("\nKvantilių palyginimas:")
    print(f"{'Kvantilis':<10}{'Sugeneruota':<15}{'Teorinis':<15}{'Skirtumas':<15}")
    
    for q in quantiles:
        idx = int(q * len(chi_squared_values))
        sample_quantile = chi_squared_values_sorted[idx]
        theoretical_quantile = stats.chi2.ppf(q, df)
        diff = abs(sample_quantile - theoretical_quantile)
        
        print(f"{q:<10}{sample_quantile:<15.4f}{theoretical_quantile:<15.4f}{diff:<15.4f}")

def plot_chi_squared_distribution(chi_squared_values, df):
    plt.figure(figsize=(10, 6))
    
    # Filtruoti ekstremalias reikšmes geresnei vizualizacijai
    max_display = np.percentile(chi_squared_values, 99)
    plot_values = [x for x in chi_squared_values if x <= max_display]
    
    # Vaizduoti histogramą
    hist, bins, _ = plt.hist(plot_values, bins=50, density=True, alpha=0.6, 
                             label='Sugeneruotas Chi-kvadrato skirstinys')
    
    # Vaizduoti teorinę chi-kvadrato tankio funkciją
    x = np.linspace(0.01, max_display, 1000)
    y = stats.chi2.pdf(x, df)
    plt.plot(x, y, 'r-', lw=2, label=f'Teorinė χ²({df}) tankio funkcija')
    
    # Diagramos nustatymai
    plt.title(f'Chi-kvadrato skirstinys su {df} laisvės laipsniais')
    plt.xlabel('Reikšmė')
    plt.ylabel('Tikimybės tankis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # LCG parametrai iš lcg_parameters_calculator.py išvesties
    a = 370  # Daugiklis (iš jūsų skaičiuotuvo)
    c = 71   # Prieaugis (iš jūsų skaičiuotuvo)
    m = 1107 # Modulis = 3^3 * 41
    seed = 1
    
    # Chi-kvadrato skirstinio parametras
    df = 5  # Laisvės laipsniai
    
    # Imties dydžio skaičiavimas
    # Kiekvienai chi-kvadrato reikšmei reikia df standartinių normaliųjų
    # Kiekviena Box-Muller transformacija reikalauja 2 tolygiai pasiskirsčiusių atsitiktinių skaičių 2 standartiniams normaliesiems sugeneruoti
    target_sample_size = 1000  # Chi-kvadrato reikšmių, kurias norime sugeneruoti, skaičius
    needed_uniforms = math.ceil(target_sample_size * (df/2))
    
    # Generuoti tolygiai pasiskirsčiusius atsitiktinius skaičius naudojant LCG
    print(f"Generuojami {needed_uniforms} tolygiai pasiskirstyti atsitiktiniai skaičiai naudojant LCG...")
    uniform_random_numbers = generate_uniform_random_numbers(a, c, m, seed, needed_uniforms)
    
    # Transformuoti į standartinį normalųjį skirstinį naudojant Box-Muller
    print("Transformuojama į standartinį normalųjį skirstinį naudojant Box-Muller transformaciją...")
    standard_normals = box_muller_transform(uniform_random_numbers)
    
    # Generuoti chi-kvadrato kintamuosius
    print(f"Generuojami chi-kvadrato kintamieji su {df} laisvės laipsniais...")
    chi_squared_values = generate_chi_squared(standard_normals, df)

    analyze_chi_squared_distribution(chi_squared_values, df)
    plot_chi_squared_distribution(chi_squared_values, df)

if __name__ == "__main__":
    main()
