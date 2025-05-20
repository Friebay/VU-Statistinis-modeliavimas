import math
import numpy as np
from lcg_parameters_calculator import generate_lcg_sequence

def monte_carlo_integration(func, a, b, num_samples=1000):
    """
    Atlikti Monte Karlo integravimą funkcijai intervale [a, b]
    
    Parametrai:
        func (function): Funkcija, kurią reikia integruoti
        a (float): Integravimo apatinė riba
        b (float): Integravimo viršutinė riba
        num_samples (int): Atsitiktinių taškų skaičius, numatytasis yra 1000
    """
    # Sugeneruojame tolygiai pasiskirsčiusius atsitiktinius skaičius naudojant LCG
    # Parametrai iš lcg_parameters_calculator išvesties
    lcg_a = 370  # Daugiklis
    lcg_c = 71   # Prieaugis
    lcg_m = 1107  # Modulis = 3^3 * 41
    seed = 1
    
    # Sugeneruojame tiksliai reikiamą imčių skaičių
    lcg_values = generate_lcg_sequence(lcg_a, lcg_c, lcg_m, seed, num_samples)
    
    # Konvertuojame į reikšmes intervale [a, b]
    uniform_samples = []
    for val in lcg_values:
        # Normalizuojame į [0, 1], tada keičiame mastelį į [a, b]
        x = a + (b - a) * (val / lcg_m)
        uniform_samples.append(x)
    
    # Įvertiname funkciją kiekviename imties taške
    function_values = [func(x) for x in uniform_samples]
    
    # Apskaičiuojame vidutinę funkcijos reikšmę ir padauginame iš intervalo pločio
    average_value = sum(function_values) / num_samples
    integral_estimate = (b - a) * average_value
    
    return integral_estimate, uniform_samples, function_values

def analytical_solution():
    """
    Apskaičiuojame analitinį integralo ∫[e iki π] (x(ln(x)+e^x))dx sprendinį
    """
    # Integralo ∫[e iki π] (x(ln(x)+e^x))dx apskaičiavimas
    # Pirmasis narys: ∫[e iki π] x ln(x) dx = [0.5x^2 ln(x) - 0.25x^2]_e^π
    # Antrasis narys: ∫[e iki π] x e^x dx reikalauja integravimo dalimis
    
    # Pirmojo nario įvertinimas viršutinėje riboje π
    term1_upper = 0.5 * (math.pi**2) * math.log(math.pi) - 0.25 * (math.pi**2)
    # Pirmojo nario įvertinimas apatinėje riboje e
    term1_lower = 0.5 * (math.e**2) * math.log(math.e) - 0.25 * (math.e**2) 
    # Pastaba: ln(e) = 1, todėl term1_lower = 0.5 * (math.e**2) - 0.25 * (math.e**2)
    
    # Antrasis narys: ∫[e iki π] x e^x dx = [x*e^x - e^x]_e^π
    term2_upper = math.pi * math.exp(math.pi) - math.exp(math.pi)
    term2_lower = math.e * math.exp(math.e) - math.exp(math.e)
    
    # Narių sujungimas
    result = (term1_upper - term1_lower) + (term2_upper - term2_lower)
    
    return result

def main():
    # Apibrėžiame integruojamą funkciją: f(x) = x(ln(x) + e^x)
    def f(x):
        return x * (math.log(x) + math.exp(x))
    
    # Integravimo ribos
    a = 1
    b = math.pi
    # Apskaičiuojame analitinį sprendinį
    exact_value = analytical_solution()
    print(f"Analitinis sprendinys: {exact_value:.10f}")
    print("\n1. Monte Karlo integravimas su tolygiuoju pasiskirstymu:")
    print(f"{'Imties dydis':<15}{'Metodas':<20}{'Įvertis':<20}{'Absoliutinė paklaida':<20}{'Santykinė paklaida (%)':<20}")
    
    # Atliekame Monte Karlo integravimą su 1000 imčių
    estimate, samples, function_values = monte_carlo_integration(f, a, b)
    abs_error = abs(estimate - exact_value)
    rel_error = abs_error / exact_value * 100
    print(f"{1000:<15}{'Tolygusis':<20}{estimate:<20.10f}{abs_error:<20.10f}{rel_error:<20.6f}")
    
    print("\nIšvada:")
    print(f"Integralas ∫[e iki π] (x(ln(x)+e^x))dx ≈ {estimate:.10f}")
    
if __name__ == "__main__":
    main()