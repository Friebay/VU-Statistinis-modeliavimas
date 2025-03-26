'''
MK užduotis 1-5
1. Sugeneruokite pseudoatsitiktiniu skaičių sekas tiesiniu kongruentiniu metodu su maksimaliu periodu, kai modulis m = 776 = 2^3 * 97 ir m = 1107 = 3^3 * 41. Daugiklius a parinkite taip, kad galingumai butu didžiausi. Prieauglio c parinkimui naudokites gretimu nariu koreliacija (teoriniai testai).
Tiesinis kongruentinis metodas: X_n+1 = (a * X_n + c) mod m, n >= 0. Čiaa skaičiai: X_0 - pradinė reikšmė, X_n - n-tas pseudoatsitiktinis skaičius, a - daugiklis, c - prieauglys, m - modulis. b = a − 1. 
Maksimalus tiesinės kongruentinės sekos periodas gaunamas, kai b = a - 1 yra visų pirminių m daliklių kartotinis ir 4 kartotinis, jei m dalijasi iš 4.
Tiesinės kongruentinės sekos su maksimaliu periodu galingumu vadinsime mažiausią natūralųjį skaičių s, kuriam b^s = 0 mod m.
'''
