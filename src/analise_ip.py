import json
import numpy as np
import matplotlib.pyplot as plt


caminho_arquivo = "../outputs./logs/train-hist.258908571" 
with open(caminho_arquivo, 'r') as f:
    history = json.load(f)

train_mse = np.array(history["train-image-loss"])
train_vgg = np.array(history["train-perceptual-loss"])

lambda_val = 1e-5
influencias = (lambda_val * train_vgg) / (train_mse + (lambda_val * train_vgg)) * 100

epocas = np.arange(1, len(influencias) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epocas, influencias, marker='o', linestyle='-', color='#d62728', linewidth=2, markersize=8)
plt.title(f'Evolução da Influência Perceptual no Treinamento lambda = {lambda_val}', fontsize=14, pad=15)
plt.xlabel('Época', fontsize=12)
plt.ylabel(r'Influência Perceptual $\Psi$ (%)', fontsize=12)
for i, txt in enumerate(influencias):
    plt.annotate(f"{txt:.4f}%", (epocas[i], influencias[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xticks(epocas)
plt.tight_layout()

plt.show()