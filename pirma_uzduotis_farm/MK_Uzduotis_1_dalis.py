import math
import fractions # Naudosime mažiausiam bendram kartotiniui (LCM)
import numpy as np  # Pridedame numpy koreliacijai skaičiuoti

# --- Pagalbinės funkcijos ---

def lcm(a, b):
  """Skaičiuoja mažiausią bendrą kartotinį (LCM)."""
  if a == 0 or b == 0:
      return 0
  return abs(a * b) // math.gcd(a, b)

def get_prime_factorization(n):
  """Grąžina skaičiaus n pirminių daliklių žodyną su jų laipsniais."""
  factors = {}
  d = 2
  temp_n = n
  while d * d <= temp_n:
    while temp_n % d == 0:
      factors[d] = factors.get(d, 0) + 1
      temp_n //= d
    d += 1
  if temp_n > 1:
    factors[temp_n] = factors.get(temp_n, 0) + 1
  return factors

def calculate_power_s(b, m):
  """
  Apskaičiuoja mažiausią natūralųjį skaičių s, kuriam b^s % m == 0.
  Naudoja m ir b pirminius daliklius.
  """
  if b == 0:
      return 1 if m != 1 else 0 # Specialus atvejis
  # Jei b ir m neturi bendrų daliklių (išskyrus 1), b^s niekada nebus 0 mod m (nebent m=1)
  if math.gcd(b, m) == 1:
      print(f"Įspėjimas: gcd(b={b}, m={m}) = 1. b^s niekada nebus 0 mod m (nebent m=1).")
      return float('inf') # Arba None, arba klaida

  m_factors = get_prime_factorization(m)
  b_factors = get_prime_factorization(b)

  max_s_needed = 0
  for p, m_power in m_factors.items():
    b_power = b_factors.get(p, 0)
    if b_power == 0:
      # Jei m pirminis daliklis p nėra b daliklis, b^s niekada nebus 0 mod m
      print(f"Klaida: m={m} pirminis daliklis {p} nėra b={b} daliklis. Negalima pasiekti b^s = 0 mod m.")
      return float('inf') # Arba None, arba klaida

    # Reikia, kad (p^b_power)^s būtų dalus iš p^m_power
    # T.y., b_power * s >= m_power
    # Mažiausias sveikas s tenkinantis tai yra ceil(m_power / b_power)
    s_needed = math.ceil(m_power / b_power)
    max_s_needed = max(max_s_needed, s_needed)

  # Mažiausias s turi tenkinti sąlygas visiems m pirminiams dalikliams
  return int(max_s_needed) # s turi būti sveikas skaičius

# --- Naujos funkcijos prieaugio c parinkimui ---

def generate_lcg_sequence(a, c, m, seed, length=1000):
    """Sugeneruoja LCG seką su nurodytais parametrais."""
    sequence = []
    x = seed
    for _ in range(length):
        x = (a * x + c) % m
        sequence.append(x / m)  # Normalizuota reikšmė [0,1)
    return np.array(sequence)

def calculate_correlation(sequence):
    """Apskaičiuoja gretimų narių koreliaciją."""
    if len(sequence) < 2:
        return 1.0  # Grąžiname didžiausią koreliaciją, jei sekos per trumpos
    # Koreliacija tarp gretimų narių
    correlation = np.corrcoef(sequence[:-1], sequence[1:])[0, 1]
    return abs(correlation)  # Grąžiname absoliučią reikšmę

def select_increment_c(a, m, seed=42, num_candidates=100):
    """
    Parenka tinkamą prieaugį c LCG generatoriui, minimizuojant gretimų narių koreliaciją.
    Tikrina, kad gcd(c, m) = 1, ir ieško mažiausios koreliacijos.
    """
    best_c = 1
    min_correlation = float('inf')
    
    # Kandidatai yra c reikšmės, kur gcd(c, m) = 1
    candidates = [c for c in range(1, min(m, 2*num_candidates), 2) if math.gcd(c, m) == 1]
    
    print(f"Analizuojami {len(candidates)} kandidatai c reikšmei...")
    
    for c in candidates:
        sequence = generate_lcg_sequence(a, c, m, seed)
        corr = calculate_correlation(sequence)
        
        if corr < min_correlation:
            min_correlation = corr
            best_c = c
    
    return best_c, min_correlation

# --- Pagrindinė programos dalis ---

# 1. Pradiniai duomenys ir modulio analizė
m = 776
print(f"--- Analizuojamas modulis m = {m} ---")

m_factors_dict = get_prime_factorization(m)
m_prime_factors = list(m_factors_dict.keys())
m_is_divisible_by_4 = (m % 4 == 0)

print(f"Pilna m faktorizacija: {m_factors_dict}")
print(f"Pirminiai m dalikliai: {m_prime_factors}")
print(f"Ar m dalijasi iš 4? {'Taip' if m_is_divisible_by_4 else 'Ne'}")

# 2. Sąlygos daugikliui 'a' (dėl maksimalaus periodo)
# Pagal Hull-Dobell teoremą, kad periodas būtų m, turi būti tenkinamos sąlygos:
# a) c ir m tarpusavyje pirminiai (gcd(c, m) = 1)
# b) b = a - 1 turi dalintis iš visų pirminių m daliklių
# c) Jei m dalijasi iš 4, tai b = a - 1 turi dalintis iš 4

print("\n--- Daugiklio 'a' parinkimas ---")
print("Sąlygos maksimaliam periodui (ilgis m):")

# Sąlyga b): b turi dalintis iš visų pirminių m daliklių
required_divisor_for_b = 1
for p in m_prime_factors:
  required_divisor_for_b = lcm(required_divisor_for_b, p)
print(f"  * b = a - 1 turi dalintis iš visų pirminių daliklių ({m_prime_factors}), taigi iš jų MBK = {required_divisor_for_b}")

# Sąlyga c): Jei m dalus iš 4, b turi dalintis iš 4
if m_is_divisible_by_4:
  print(f"  * Kadangi m ({m}) dalijasi iš 4, b = a - 1 turi dalintis ir iš 4.")
  required_divisor_for_b = lcm(required_divisor_for_b, 4)
else:
  print(f"  * Kadangi m ({m}) nesidalija iš 4, papildomos sąlygos dėl dalumo iš 4 nėra.")

print(f"Apibendrinus, b = a - 1 turi būti {required_divisor_for_b} kartotinis.")

# 3. Parenkame 'a' ir apskaičiuojame 'b'
# Renkamės mažiausią galimą b = a - 1, kuris tenkina sąlygas.
# Tai bus pats required_divisor_for_b.
b = required_divisor_for_b
a = b + 1

# Patikrinam, ar 0 < a < m
if not (0 < a < m):
    print(f"Klaida: Gautas a={a} nėra intervale (0, {m}). Reikia rinktis kitą kartotinį.")
    # Čia būtų galima rinktis b = 2 * required_divisor_for_b ir t.t.
    # Bet dažniausiai mažiausias tinka.
else:
    print(f"\nParenkame mažiausią tinkamą b = a - 1 reikšmę: b = {b}")
    print(f"Tada daugiklis: a = b + 1 = {a}")

# 4. Apskaičiuojame "galingumą" s
# s yra mažiausias natūralus skaičius, kuriam b^s ≡ 0 mod m
print("\n--- Galingumo 's' skaičiavimas ---")
print(f"Ieškome mažiausio natūralaus s, kuriam b^s ≡ 0 mod m, kai b = {b}, m = {m}.")

s = calculate_power_s(b, m)

if s is None or s == float('inf'):
    print("Klaida skaičiuojant galingumą s.")
else:
    print(f"Apskaičiuotas galingumas: s = {s}")

# 5. Apskaičiuojame prieaugį c
print("\n--- Prieaugio 'c' parinkimas ---")
print("Ieškome c, kad gcd(c, m) = 1 ir minimizuotų gretimų narių koreliaciją.")

seed = 42  # Pradinis sėklos skaičius testavimui
c, correlation = select_increment_c(a, m, seed)

print(f"Parinkta prieaugio reikšmė: c = {c}")
print(f"Gretimų narių koreliacija: {correlation:.6f}")
print(f"\nGalutiniai LCG parametrai:")
print(f"X_(n+1) = ({a} * X_n + {c}) mod {m}")

# Patikriname, ar gcd(c, m) = 1
print(f"gcd(c, m) = {math.gcd(c, m)} (turėtų būti 1)")