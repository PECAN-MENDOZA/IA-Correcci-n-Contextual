"""
domain/services/noise_injector.py
Servicio de dominio: Inyecta ruido semántico, gramatical y ortográfico
para entrenar la red neuronal (T5) en corrección de contexto.
"""
import random
import re

class NoiseInjectorService:
    def __init__(self):
        # 1. Errores de contexto (Homófonos)
        self.homophones = {
            "tuvo": ["tubo"], "tubo": ["tuvo"],
            "si no": ["sino"], "sino": ["si no"],
            "vaya": ["valla", "baya"], "halla": ["haya", "allá"],
            "a ver": ["haber"], "haber": ["a ver"],
            "ay": ["hay", "ahí"], "ahí": ["ay", "hay"], "hay": ["ay", "ahí"]
        }
        # 2. Conectores para omitir
        self.connectors = [" el ", " la ", " los ", " las ", " de la ", " del ", " al ", " un ", " una "]

    def inject(self, text: str) -> str:
        noisy = text

        # A) Romper concordancia (Singular/Plural)
        if "fueron" in noisy.lower() and random.random() < 0.5:
            noisy = re.sub(r'\bfueron\b', 'fue', noisy, flags=re.IGNORECASE)
        elif "eran" in noisy.lower() and random.random() < 0.5:
            noisy = re.sub(r'\beran\b', 'era', noisy, flags=re.IGNORECASE)

        if "había" in noisy.lower() and random.random() < 0.3:
            noisy = re.sub(r'\bhabía\b', 'habían', noisy, flags=re.IGNORECASE)

        # B) Omisión de conectores
        if random.random() < 0.4:
            for conn in self.connectors:
                if conn in noisy.lower():
                    noisy = re.sub(rf'(?i){conn}', ' ', noisy, count=1)
                    break

        # C) Intercambio de Homófonos destructivo
        if random.random() < 0.4:
            for correct, wrongs in self.homophones.items():
                if re.search(rf'\b{correct}\b', noisy, flags=re.IGNORECASE):
                    wrong_choice = random.choice(wrongs)
                    noisy = re.sub(rf'\b{correct}\b', wrong_choice, noisy, count=1, flags=re.IGNORECASE)

        # D) Ruido ortográfico fonético
        if random.random() < 0.5:
            tildes = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u'}
            for acento, sin_acento in tildes.items():
                noisy = noisy.replace(acento, sin_acento)
            noisy = noisy.replace("ll", "y").replace("qu", "k").replace("c", "s")

        return ' '.join(noisy.split())