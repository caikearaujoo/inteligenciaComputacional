"""
modelo.py - MLP para classificacao de cobertura vegetal (Forest Covertype)
Disciplina: Inteligencia Computacional
Autores: Caike e Izidro

Constroi, treina e avalia a rede neural sobre os dados preparados em
preprocessamento.py (split 70/15/15 + phi(X) de terreno + normalizacao).

Ativacao e inicializacao foram decididas por experimentos (experimentos.py),
sempre pela VALIDACAO: tanh, com o tamanho dos pesos calibrado pelo metodo do
Desafio 2 (GBC073), sigma^2 = 1 / (fan_in * E[f(z)^2]), em que E[f(z)^2] e
estimado por amostragem Monte Carlo (serve para qualquer ativacao, sem decorar
a formula fechada de cada uma). A reducao de escala da ultima camada (x0,5,
do exemplo do professor) foi retirada porque nao mostrou efeito. O ReLU, que
era o padrao antes, ficou em ultimo lugar entre as ativacoes testadas.

Otimizador e agenda de taxa de aprendizado foram decididos no Desafio 3
(desafio3/, protocolo completo e diagnostico em desafio3/desafio3.md): MUON
(lr 0,03) nas 2 matrizes ocultas, AdamW (lr 0,01) nos vieses e na camada de
saida, com agenda WARMUP (5%) + COSSENO ate 0, 40 epocas, semente 0. E a receita
vencedora, validada com 3 sementes (F1 de validacao 0,9121 +- 0,0009 contra
0,8646 +- 0,0028 da baseline Adam 1e-3 constante). Rodar este arquivo reproduz,
com a semente 0, o mesmo modelo que o desafio3/teste_final.py avaliou no teste.

Detalhe de PyTorch: a rede devolve LOGITS crus na ultima camada, sem
Softmax. O F.cross_entropy ja aplica o Softmax por dentro (de forma
numericamente mais estavel). Colocar um Softmax antes dele aplicaria a
funcao duas vezes e atrapalharia o treino. O Softmax so e necessario se
quisermos as probabilidades na hora de usar o modelo.

Regra de uso dos conjuntos: o treino ajusta os pesos, a VALIDACAO decide
(epoca, arquitetura, ativacao, inicializacao...), e o TESTE so e olhado no final.
"""

import copy
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
from preprocessamento import NOMES_CLASSES, executar_pipeline

# 57 entradas -> 128 -> 64 -> 7 saidas = 16.135 parametros treinaveis.
# Com 406.708 exemplos de treino, sao ~25 exemplos por parametro, dentro da
# regra pratica de ~20 discutida em aula.
CAMADAS_OCULTAS = (128, 64)
# 40 epocas = 31.800 passos: o orcamento fixo do protocolo do Desafio 3.
EPOCAS = 40
ESCALA_ULTIMA = 1.0     # 0.5 era o ajuste do exemplo do professor; sem efeito medido
TAMANHO_LOTE = 512

# Receita vencedora do Desafio 3
TAXA_MUON = 0.03        # matrizes ocultas
TAXA_ADAMW = 0.01       # vieses e camada de saida
PESO_DECAY = 0.01       # weight decay (Muon e AdamW)
FRACAO_WARMUP = 0.05    # primeiros 5% dos passos: o lr sobe de ~0 ao valor alvo

SEMENTE = 0


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


# E[f(z)^2] com z ~ N(0,1), estimado uma vez por amostragem (Monte Carlo)
_gerador_calibracao = torch.Generator().manual_seed(0)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_gerador_calibracao)).pow(2).mean().item()


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                 fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    desvio = math.sqrt(1.0 / (fan_in * _E_f2))
    if camada == n_camadas:
        desvio *= ESCALA_ULTIMA
    W.normal_(0.0, desvio)
    b.zero_()


class MLP(nn.Module):
    """
    MLP: Linear -> ativacao -> ... -> Linear (logits).
    """

    def __init__(self, d_entrada, n_classes, camadas_ocultas=CAMADAS_OCULTAS,
                 ativ=ativacao, init=inicializar):
        # ativ e init podem ser trocados nos experimentos. init=None deixa a
        # inicializacao padrao do PyTorch; camadas_ocultas=() da um modelo
        # linear (sem camada oculta).
        super().__init__()
        dims = [d_entrada, *camadas_ocultas, n_classes]
        self.camadas = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.ativ = ativ

        n_camadas = len(self.camadas)
        if init is not None:
            with torch.no_grad():
                for k, linear in enumerate(self.camadas, start=1):
                    init(linear.weight, linear.bias, linear.in_features,
                         linear.out_features, k, n_camadas)

    def forward(self, x):
        for k, linear in enumerate(self.camadas):
            x = linear(x)
            if k < len(self.camadas) - 1:   # ultima camada devolve logits crus
                x = self.ativ(x)
        return x


def prever(modelo, X):
    modelo.eval()
    with torch.no_grad():
        return modelo(X).argmax(dim=1).numpy()


def criar_otimizadores(modelo):
    """
    Devolve [Muon, AdamW]. O Muon so vale para matrizes 2-D das camadas
    OCULTAS (ele ortogonaliza o momento: todas as direcoes singulares recebem o
    mesmo tamanho de passo). Os vieses e a camada de saida ficam no AdamW.
    """
    n_camadas = len(modelo.camadas)
    ocultas = []
    resto = []
    for nome, p in modelo.named_parameters():
        indice_camada = int(nome.split(".")[1])
        if p.ndim == 2 and indice_camada < n_camadas - 1:
            ocultas.append(p)
        else:
            resto.append(p)
    muon = torch.optim.Muon(ocultas, lr=TAXA_MUON, momentum=0.95, nesterov=True,
                            weight_decay=PESO_DECAY, ns_steps=5)
    adamw = torch.optim.AdamW(resto, lr=TAXA_ADAMW, weight_decay=PESO_DECAY)
    return [muon, adamw]


def treinar(modelo, X_treino, y_treino, X_val, y_val, epocas=EPOCAS,
            tamanho_lote=TAMANHO_LOTE, semente=SEMENTE):
    """
    Treina em mini-lotes com Muon + AdamW e agenda warmup + cosseno e, a cada
    epoca, mede a VALIDACAO. Devolve o modelo na melhor epoca segundo o F1 macro
    da validacao - o F1 macro e a metrica escolhida porque a acuracia simples
    favorece as classes grandes, e o F1 macro trata todas as 7 por igual.
    """
    otimizadores = criar_otimizadores(modelo)

    # agenda por PASSO: sobe linear em 5% dos passos (warmup) e depois cai como
    # um cosseno ate 0. O fator multiplica o lr de cada otimizador.
    total_passos = epocas * math.ceil(len(X_treino) / tamanho_lote)
    aquecimento = max(1, int(FRACAO_WARMUP * total_passos))

    def fator_agenda(passo):
        if passo < aquecimento:
            return (passo + 1) / aquecimento
        progresso = (passo - aquecimento) / (total_passos - aquecimento)
        return 0.5 * (1.0 + math.cos(math.pi * progresso))

    agendas = [torch.optim.lr_scheduler.LambdaLR(o, fator_agenda) for o in otimizadores]
    gerador = torch.Generator().manual_seed(semente)

    melhor_f1, melhor_epoca, melhor_estado = -1.0, 0, None
    for epoca in range(1, epocas + 1):
        modelo.train()
        perda_total = 0.0
        # a cada epoca embaralha os indices e da um passo a cada `tamanho_lote`
        for idx in torch.randperm(len(X_treino), generator=gerador).split(tamanho_lote):
            for o in otimizadores:
                o.zero_grad()
            perda = F.cross_entropy(modelo(X_treino[idx]), y_treino[idx])
            perda.backward()
            for o in otimizadores:
                o.step()
            for a in agendas:
                a.step()
            perda_total += perda.item() * len(idx)

        predito_val = prever(modelo, X_val)
        f1_val = f1_score(y_val.numpy(), predito_val, average="macro", zero_division=0)
        acc_val = accuracy_score(y_val.numpy(), predito_val)

        if f1_val > melhor_f1:
            melhor_f1, melhor_epoca = f1_val, epoca
            melhor_estado = copy.deepcopy(modelo.state_dict())

        if epoca % 20 == 0 or epoca == 1:
            print(f"  epoca {epoca:3d}/{epocas} - perda treino: {perda_total / len(y_treino):.4f}"
                  f" | validacao: acc {acc_val:.4f}, F1 macro {f1_val:.4f}")

    modelo.load_state_dict(melhor_estado)
    print(f"  melhor epoca (F1 macro da validacao): {melhor_epoca} ({melhor_f1:.4f})")
    return modelo


def avaliar(modelo, X, y, nome_conjunto):
    y_real = y.numpy()
    predito = prever(modelo, X)

    acc = accuracy_score(y_real, predito)
    trivial = np.bincount(y_real).max() / len(y_real)

    print(f"\n--- {nome_conjunto} ---")
    print(f"Acuracia simples   : {acc:.4f}   (chutar a majoritaria: {trivial:.4f})")
    print(f"Acuracia balanceada: {balanced_accuracy_score(y_real, predito):.4f}"
          f"  (media do recall por classe)")
    print(f"F1 macro           : {f1_score(y_real, predito, average='macro', zero_division=0):.4f}")
    print(f"\nRelatorio por classe ({nome_conjunto}):")
    print(classification_report(y_real, predito, target_names=NOMES_CLASSES,
                                digits=3, zero_division=0))
    return predito, confusion_matrix(y_real, predito)


def plotar_matriz_confusao(matriz, caminho_saida="matriz_confusao_teste.png"):
    """
    Matriz de confusao 7x7 do teste, normalizada por linha (cada linha soma 1):
    mostra, para cada cobertura real, com o que a rede a confunde.
    """
    normalizada = matriz.astype(float) / matriz.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(NOMES_CLASSES)))
    ax.set_yticks(range(len(NOMES_CLASSES)))
    ax.set_xticklabels(NOMES_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(NOMES_CLASSES, fontsize=9)
    for i in range(len(NOMES_CLASSES)):
        for j in range(len(NOMES_CLASSES)):
            ax.text(j, i, f"{normalizada[i, j]:.2f}", ha="center", va="center",
                    color="white" if normalizada[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusao (teste, normalizada por linha)")
    fig.colorbar(im, ax=ax, label="Proporcao")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    print(f"\nMatriz de confusao salva em: {caminho_saida}")


def main():
    torch.manual_seed(SEMENTE)

    X_tr, X_va, X_te, y_tr, y_va, y_te, scaler = executar_pipeline()

    X_tr_t = torch.tensor(X_tr.to_numpy(), dtype=torch.float32)
    X_va_t = torch.tensor(X_va.to_numpy(), dtype=torch.float32)
    X_te_t = torch.tensor(X_te.to_numpy(), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_va_t = torch.tensor(y_va, dtype=torch.long)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    n_classes = len(NOMES_CLASSES)
    modelo = MLP(d_entrada=X_tr_t.shape[1], n_classes=n_classes)

    total_parametros = sum(p.numel() for p in modelo.parameters())
    camadas = " -> ".join(str(c) for c in CAMADAS_OCULTAS)
    print(f"\nArquitetura: {X_tr_t.shape[1]} -> {camadas} -> {n_classes}")
    print(f"Total de parametros: {total_parametros}")
    print(f"Exemplos de treino por parametro: {len(X_tr_t) / total_parametros:.1f}")

    print(f"\nTreinando por {EPOCAS} epocas...")
    treinar(modelo, X_tr_t, y_tr_t, X_va_t, y_va_t)

    avaliar(modelo, X_va_t, y_va_t, "validacao")
    _predito_teste, matriz_teste = avaliar(modelo, X_te_t, y_te_t, "teste (olhado so agora)")
    plotar_matriz_confusao(matriz_teste)


if __name__ == "__main__":
    main()
