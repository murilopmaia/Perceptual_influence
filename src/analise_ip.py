import json
import numpy as np
import matplotlib.pyplot as plt

caminho_arquivo = "../outputs/logs/train-hist.957508344" 
n_epocas = 23
with open(caminho_arquivo, 'r') as f:
    history = json.load(f)

train_mse = np.array(history["train-image-loss"])
train_vgg = np.array(history["train-perceptual-loss"])

lambda_used = np.array(history["lambda-used"])
lambda_calculated = np.array(history["lambda-calculated"])

if n_epocas is not None:
    train_mse = train_mse[:n_epocas]
    train_vgg = train_vgg[:n_epocas]
    lambda_used = lambda_used[:n_epocas]
    lambda_calculated = lambda_calculated[:n_epocas]


influencia_usada = (lambda_used * train_vgg) / (train_mse + (lambda_used * train_vgg)) * 100


influencia_calculada = (lambda_calculated * train_vgg) / (train_mse + (lambda_calculated * train_vgg)) * 100

epocas = np.arange(1, len(influencia_usada) + 1)

plt.figure(figsize=(15, 7))

plt.plot(epocas, influencia_usada, marker='o', linestyle='-', color='#d62728', 
         linewidth=2, markersize=8, label='Influência Real (Lambda Utilizado)')

plt.plot(epocas, influencia_calculada, marker='s', linestyle='--', color='#1f77b4', 
         linewidth=2, markersize=8, label='Influência Alvo (Lambda Calculado)')


plt.axhline(y=95.0, color='gray', linestyle=':', linewidth=2, label='Meta (95%)')

plt.title('Convergência da Influência Perceptual (Real vs. Ajuste)', fontsize=15, pad=15)
plt.xlabel('Época', fontsize=12)
plt.ylabel(r'Influência Perceptual $\Psi$ (%)', fontsize=12)


for i, txt in enumerate(influencia_usada):
    plt.annotate(f"{txt:.2f}%", 
                 (epocas[i], influencia_usada[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='#d62728')

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xticks(epocas)
plt.legend(fontsize=11)
plt.ylim(89, 101)
plt.tight_layout()

plt.show()