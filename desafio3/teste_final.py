"""
teste_final.py - avaliacao no conjunto de TESTE, feita UMA UNICA VEZ (Desafio 3)

Avalia no teste dois modelos treinados so com treino e validacao:
  - a receita vencedora (Muon, lr 0,03, warmup + cosseno, semente 0, 40 epocas)
  - a baseline (Adam, lr 1e-3, constante, semente 0, 40 epocas)
Os pesos sao os da melhor epoca pelo F1 da VALIDACAO (salvos por treino.py --salvar).

Regra do enunciado: "teste usado uma vez". Este script grava teste_final.txt e se
RECUSA a rodar de novo se o arquivo existir (para nao ficar tentado a ajustar
nada olhando o teste).

Uso (da pasta desafio3):  py teste_final.py
"""

import contextlib
import io
import os
import sys

import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             f1_score)

PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PASTA, "..", "trabalhoFinal"))
from modelo import MLP                                       # noqa: E402
from preprocessamento import NOMES_CLASSES, executar_pipeline  # noqa: E402

SAIDA = os.path.join(PASTA, "teste_final.txt")
MODELOS = [
    ("Receita vencedora: Muon lr 0,03, warmup + cosseno", "final_muon_s0_e40"),
    ("Baseline: Adam lr 1e-3, constante", "final_baseline_s0_e40"),
]

if os.path.exists(SAIDA):
    print("teste_final.txt ja existe: o teste ja foi usado. Nada a fazer.")
    sys.exit(1)

with contextlib.redirect_stdout(io.StringIO()):
    _tr, _va, X_te, _ytr, _yva, y_te, _ = executar_pipeline()
X = torch.tensor(X_te.to_numpy(), dtype=torch.float32)
y = y_te

linhas = []
for rotulo, nome in MODELOS:
    modelo = MLP(X.shape[1], len(NOMES_CLASSES))
    modelo.load_state_dict(torch.load(os.path.join(PASTA, "logs", f"{nome}.pt")))
    modelo.eval()
    with torch.no_grad():
        predito = modelo(X).argmax(dim=1).numpy()
    linhas.append(f"=== {rotulo} ({len(y)} exemplos de teste) ===")
    linhas.append(f"Acuracia simples   : {accuracy_score(y, predito):.4f}")
    linhas.append(f"Acuracia balanceada: {balanced_accuracy_score(y, predito):.4f}")
    linhas.append(f"F1 macro           : {f1_score(y, predito, average='macro', zero_division=0):.4f}")
    linhas.append(classification_report(y, predito, target_names=NOMES_CLASSES, digits=3, zero_division=0))

texto = "\n".join(linhas)
print(texto)
with open(SAIDA, "w", encoding="utf-8") as arq:
    arq.write(texto + "\n")
print("salvo em", SAIDA)
