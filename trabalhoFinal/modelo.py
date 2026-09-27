"""
modelo.py - MLP para o Sistema de Recomendacao de Culturas Agricolas
Disciplina: Inteligencia Computacional
Autores: Caike e Izidro

Constroi, treina e avalia a rede neural sobre os dados preparados em
preprocessamento.py (limpeza + SMOTENC + phi(X) + normalizacao + codificacao).

Dataset: dados REAIS da Etiopia (Mendeley 10.17632/8v757rr4st.1) - 3.867
registros, 12 culturas, com desbalanceamento real (Teff 1.260 x Fallow 26).
Por isso a avaliacao reporta tambem acuracia BALANCEADA e F1 macro, nao so
acuracia simples: com classes desbalanceadas, acertar so a classe majoritaria
ja daria uma acuracia simples enganosamente alta.

Ativacao e inicializacao reaproveitam a mesma tecnica do Desafio 2
(GBC073): ReLU, com o tamanho dos pesos calibrado por amostragem Monte
Carlo (em vez de decorar a formula fechada do He) e a escala reduzida na
ultima camada (mesmo ajuste usado no exemplo_submissao_d2.py do
professor, inspirado no conceito de Fixup Initialization). Aqui a rede e
rasa (2 camadas ocultas), entao esse cuidado nao e estritamente necessario
como era no Desafio 2 (48 camadas) - mantemos por continuidade e porque o
codigo ja estava validado.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from preprocessamento import executar_pipeline

CAMINHO_DATASET = "dados/Crop_Ethiopia_soil_weather.csv"
CAMADAS_OCULTAS = (32, 16)
EPOCAS = 50
TAXA_APRENDIZADO = 1e-3
TAMANHO_LOTE = 64
SEMENTE = 42


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


_gerador_calibracao = torch.Generator().manual_seed(0)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_gerador_calibracao)).pow(2).mean().item()


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                 fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    desvio = math.sqrt(1.0 / (fan_in * _E_f2))
    if camada == n_camadas:
        desvio *= 0.5
    W.normal_(0.0, desvio)
    b.zero_()


class MLP(nn.Module):
    """MLP simples: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear (logits)."""

    def __init__(self, d_entrada, n_classes, camadas_ocultas=CAMADAS_OCULTAS):
        super().__init__()
        dims = [d_entrada, *camadas_ocultas, n_classes]
        self.camadas = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        n_camadas = len(self.camadas)
        with torch.no_grad():
            for k, linear in enumerate(self.camadas, start=1):
                inicializar(linear.weight, linear.bias, linear.in_features,
                            linear.out_features, k, n_camadas)

    def forward(self, x):
        for k, linear in enumerate(self.camadas):
            x = linear(x)
            if k < len(self.camadas) - 1:
                x = ativacao(x)
        return x


def treinar(modelo, X_treino, y_treino, epocas=EPOCAS, lr=TAXA_APRENDIZADO,
            tamanho_lote=TAMANHO_LOTE, semente=SEMENTE):
    """
    Treina em mini-lotes: em vez de olhar os 6.655 exemplos de uma vez e dar
    um unico passo por epoca, o treino embaralha os dados e da um passo a cada
    `tamanho_lote` exemplos. Isso multiplica o numero de atualizacoes de peso
    por epoca e e a pratica padrao - com lote completo a rede aprendia devagar
    demais neste dataset.
    """
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    gerador = torch.Generator().manual_seed(semente)

    for epoca in range(1, epocas + 1):
        modelo.train()
        perda_epoca = 0.0
        indices = torch.randperm(len(X_treino), generator=gerador)
        for lote in indices.split(tamanho_lote):
            otimizador.zero_grad()
            perda = F.cross_entropy(modelo(X_treino[lote]), y_treino[lote])
            perda.backward()
            otimizador.step()
            perda_epoca += perda.item() * len(lote)

        if epoca % 10 == 0 or epoca == 1:
            print(f"  epoca {epoca:3d}/{epocas} - perda media: {perda_epoca/len(X_treino):.4f}")
    return modelo


def avaliar(modelo, X, y, nomes_classes, nome_conjunto):
    modelo.eval()
    with torch.no_grad():
        logits = modelo(X)
        predito = logits.argmax(dim=1).numpy()
    y_real = y.numpy()

    acc = accuracy_score(y_real, predito)
    acc_balanceada = balanced_accuracy_score(y_real, predito)
    f1_macro = f1_score(y_real, predito, average="macro", zero_division=0)

    print(f"\n--- {nome_conjunto} ---")
    print(f"Acuracia simples   : {acc:.4f}")
    print(f"Acuracia balanceada: {acc_balanceada:.4f}  (media do acerto POR CLASSE)")
    print(f"F1 macro           : {f1_macro:.4f}  (equilibra precisao e recall entre classes)")
    print(f"\nRelatorio de classificacao ({nome_conjunto}):")
    print(classification_report(y_real, predito, target_names=nomes_classes, zero_division=0))

    return acc, confusion_matrix(y_real, predito)


def plotar_matriz_confusao(matriz, nomes_classes, caminho_saida="matriz_confusao_teste.png"):
    """
    Salva a matriz de confusao (conjunto de teste) como imagem, normalizada
    por linha (recall visual): cada linha soma 1, facilita ver com o que
    cada cultura real esta sendo confundida.
    """
    matriz_normalizada = matriz.astype(float) / matriz.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(matriz_normalizada, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(nomes_classes)))
    ax.set_yticks(range(len(nomes_classes)))
    ax.set_xticklabels(nomes_classes, rotation=90, fontsize=8)
    ax.set_yticklabels(nomes_classes, fontsize=8)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusao (teste, normalizada por linha)")
    fig.colorbar(im, ax=ax, label="Proporcao")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    print(f"\nMatriz de confusao salva em: {caminho_saida}")


def main():
    torch.manual_seed(SEMENTE)

    X_treino, X_teste, y_treino, y_teste, scaler, encoder_rotulo = executar_pipeline(CAMINHO_DATASET)

    X_treino_t = torch.tensor(X_treino.to_numpy(), dtype=torch.float32)
    X_teste_t = torch.tensor(X_teste.to_numpy(), dtype=torch.float32)
    y_treino_t = torch.tensor(y_treino, dtype=torch.long)
    y_teste_t = torch.tensor(y_teste, dtype=torch.long)

    n_classes = len(encoder_rotulo.classes_)
    modelo = MLP(d_entrada=X_treino_t.shape[1], n_classes=n_classes)

    print(f"\nArquitetura: {X_treino_t.shape[1]} -> {' -> '.join(str(c) for c in CAMADAS_OCULTAS)} -> {n_classes}")
    total_parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Total de parametros: {total_parametros}")

    print(f"\nTreinando por {EPOCAS} epocas...")
    treinar(modelo, X_treino_t, y_treino_t)

    avaliar(modelo, X_treino_t, y_treino_t, encoder_rotulo.classes_, "treino")
    _, matriz_teste = avaliar(modelo, X_teste_t, y_teste_t, encoder_rotulo.classes_, "teste")
    plotar_matriz_confusao(matriz_teste, encoder_rotulo.classes_)


if __name__ == "__main__":
    main()
