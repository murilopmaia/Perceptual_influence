import numpy as np


MSE = 0.0004745166515931487
VGG_loss = 2113.844482421875
INFLUENCIA_ALVO = 95.0


# METODO 1: 
lambdas_testes = np.logspace(-7, 0, 10000)
influencias = (lambdas_testes * VGG_loss / (MSE + (lambdas_testes * VGG_loss))) * 100
indice_ideal = np.abs(influencias - INFLUENCIA_ALVO).argmin()
lambda_aprox = lambdas_testes[indice_ideal]


# METODO 2: Isolamento 
T = INFLUENCIA_ALVO / 100.0
lambda_exato = (T * MSE) / (VGG_loss * (1 - T))

# COMPARACAO
print(f"Metodo 1 (Aproximacao por Array): {lambda_aprox:.8e}")
print(f"Metodo 2 (Calculo Algebrico Exato): {lambda_exato:.8e}")
print(f"Diferenca absoluta: {abs(lambda_exato - lambda_aprox):.8e}")