import sys
from math import gcd

def lcg(a, c, m, seed, n):
    """
    Sugeneruoja pseudoatsitiktinių skaičių seką naudojant linijinį kongruentinį metodą.
    
    a: daugiklis
    m: modulis
    seed: pradinė reikšmė X0
    n: sugeneruojamų skaičių kiekis
    
    Grąžina: sugeneruotų skaičių sąrašą.
    """
    sequence = []
    x = seed
    try:
        for _ in range(n):
            x = (a * x + c) % m
            sequence.append(x)
        return sequence
    except Exception as e:
        print("Klaida generuojant seką:", e)
        sys.exit(1)

def compute_galingumas(b, m):
    """
    Apskaičiuoja linijinės kongruentinės sekos galingumą –
    mažiausią natūralų s, kuriam b^s mod m = 0.
    
    b: a - 1
    m: modulis
    
    Grąžina: galingumo laipsnį s.
    """
    s = 1
    try:
        # Naudojame iteraciją, kol b^s mod m tampa lygus 0.
        while True:
            if pow(b, s, m) == 0:
                return s
            s += 1
            # Sauga: jei s tampa per didelis – nutraukiam.
            if s > 100:
                raise ValueError("Galingumas nerastas per 100 iteracijų")
    except Exception as e:
        print("Klaida apskaičiuojant galingumą:", e)
        sys.exit(1)

def main():
    try:
        # Modulis m = 776 = 2^3 * 97
        m = 776
        
        # Parenkamas multiplikatorius a. Sąlyga: b = a - 1 turi būti dalus iš 4 ir 97 (t. y. iš 388)
        # Pasirenkame a = 389, todėl b = 388 = 4 * 97, kas užtikrina maksimalų periodą.
        a = 389
        b = a - 1
        
        # Išvedame visus parametro nustatymus
        print("Parametrai:")
        print(f"m (modulis): {m}")
        print(f"a (daugiklis): {a}")
        print(f"b (a - 1): {b}")
        
        # Apskaičiuojame sekos galingumą.
        galingumas = compute_galingumas(b, m)
        print(f"Galingumas (power) yra: {galingumas}")
        
        
    except Exception as e:
        print("Bendra klaida:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()