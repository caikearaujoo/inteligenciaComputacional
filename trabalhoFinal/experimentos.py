"""
experimentos.py - Experimentos do trabalho final
Disciplina: Inteligencia Computacional
Autores: Caike e Izidro

Compara, no Forest Covertype:
  - phi (Desafio 1): quais termos do mapa de caracteristicas ajudam;
  - ativacao e inicializacao (Desafio 2): qual combinacao funciona melhor.

REGRA: estes experimentos usam SOMENTE a validacao. O conjunto de teste nem
e carregado em tensor aqui; ele so e olhado uma vez, no final, com a
configuracao escolhida (ver modelo.py).

Cada configuracao roda com varias sementes (media +- desvio padrao), porque
quando as diferencas sao pequenas uma semente so nao separa efeito real de
sorte de inicializacao.

Uso:
    py -u experimentos.py phi
    py -u experimentos.py ativacao
    py -u experimentos.py phi ativacao
    py -u experimentos.py confirmacao
"""

import contextlib
import csv
import io
import math
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from modelo import MLP
from preprocessamento import executar_pipeline

ARQUIVO_RESULTADOS = "resultados_experimentos.csv"
N_CLASSES = 7
LOTE = 512
TAXA_APRENDIZADO = 1e-3
CAMADAS_PADRAO = (128, 64)
TERMOS_COMPLETO = ("aspecto", "agua", "elev_decl")


# ---------------------------------------------------------------------------
# dados (so treino e validacao)
# ---------------------------------------------------------------------------
_cache_dados = {}


def dados(termos_phi):
    chave = tuple(termos_phi)
    if chave not in _cache_dados:
        with contextlib.redirect_stdout(io.StringIO()):
            X_tr, X_va, _teste, y_tr, y_va, _y_teste, _ = executar_pipeline(termos_phi)
        _cache_dados[chave] = (
            torch.tensor(X_tr.to_numpy()), torch.tensor(y_tr),
            torch.tensor(X_va.to_numpy()), torch.tensor(y_va),
        )
    return _cache_dados[chave]


# ---------------------------------------------------------------------------
# ativacoes e inicializacoes
# ---------------------------------------------------------------------------
ATIVACOES = {
    "relu": F.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, 0.01),
    "gelu": F.gelu,
    "selu": F.selu,
    "tanh": torch.tanh,
}


def ganho_quadrado(ativ):
    """E[f(z)^2] com z ~ N(0,1), por Monte Carlo (mesmo metodo do Desafio 2)."""
    gerador = torch.Generator().manual_seed(0)
    return ativ(torch.randn(1_000_000, generator=gerador)).pow(2).mean().item()


def init_calibrada(ativ, escala_ultima):
    """Desafio 2: sigma^2 = 1 / (fan_in * E[f(z)^2]); ultima camada x escala."""
    e_f2 = ganho_quadrado(ativ)

    @torch.no_grad()
    def init(W, b, fan_in, fan_out, camada, n_camadas):
        desvio = math.sqrt(1.0 / (fan_in * e_f2))
        if camada == n_camadas:
            desvio *= escala_ultima
        W.normal_(0.0, desvio)
        b.zero_()
    return init


@torch.no_grad()
def init_lecun(W, b, fan_in, fan_out, camada, n_camadas):
    W.normal_(0.0, math.sqrt(1.0 / fan_in))
    b.zero_()


@torch.no_grad()
def init_xavier(W, b, fan_in, fan_out, camada, n_camadas):
    nn.init.xavier_normal_(W)
    b.zero_()


# cada entrada recebe a ativacao e devolve a funcao de inicializacao
# (None = inicializacao padrao do PyTorch, kaiming uniforme)
INICIALIZACOES = {
    "calibrada_x0.5": lambda a: init_calibrada(a, 0.5),   # a do Desafio 2
    "calibrada_x1": lambda a: init_calibrada(a, 1.0),     # sem reduzir a ultima
    "lecun": lambda a: init_lecun,
    "xavier": lambda a: init_xavier,
    "pytorch_padrao": lambda a: None,
}


# ---------------------------------------------------------------------------
# treino e medicao
# ---------------------------------------------------------------------------
def treinar_e_medir(modelo, X_tr, y_tr, X_va, y_va, epocas, semente):
    """Treina (Adam, lr fixo) e devolve as metricas da melhor epoca (F1 macro da validacao)."""
    otimizador = torch.optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)
    gerador = torch.Generator().manual_seed(semente)
    y_val = y_va.numpy()
    melhor = (-1.0, 0.0, 0.0, 0)

    for epoca in range(1, epocas + 1):
        modelo.train()
        for idx in torch.randperm(len(X_tr), generator=gerador).split(LOTE):
            otimizador.zero_grad()
            F.cross_entropy(modelo(X_tr[idx]), y_tr[idx]).backward()
            otimizador.step()

        modelo.eval()
        with torch.no_grad():
            predito = modelo(X_va).argmax(dim=1).numpy()
        f1 = f1_score(y_val, predito, average="macro", zero_division=0)
        if f1 > melhor[0]:
            melhor = (f1, accuracy_score(y_val, predito),
                      balanced_accuracy_score(y_val, predito), epoca)
    return melhor


_resultados = []


def rodar(experimento, config, termos_phi, camadas, ativ, init, epocas, semente):
    X_tr, y_tr, X_va, y_va = dados(termos_phi)
    torch.manual_seed(semente)
    modelo = MLP(X_tr.shape[1], N_CLASSES, camadas_ocultas=camadas, ativ=ativ, init=init)

    inicio = time.time()
    f1, acc, bal, melhor_epoca = treinar_e_medir(modelo, X_tr, y_tr, X_va, y_va, epocas, semente)
    segundos = time.time() - inicio

    linha = dict(experimento=experimento, config=config, semente=semente, epocas=epocas,
                 f1_macro=round(f1, 4), acuracia=round(acc, 4), balanceada=round(bal, 4),
                 melhor_epoca=melhor_epoca, segundos=round(segundos, 1))
    _resultados.append(linha)
    with open(ARQUIVO_RESULTADOS, "a", newline="", encoding="utf-8") as arq:
        escritor = csv.DictWriter(arq, fieldnames=list(linha.keys()))
        if arq.tell() == 0:
            escritor.writeheader()
        escritor.writerow(linha)

    print(f"  [{experimento}] {config:<34} semente {semente} | F1 {f1:.4f} | "
          f"acc {acc:.4f} | melhor epoca {melhor_epoca:2d} | {segundos:5.1f}s", flush=True)
    return linha


def resumir(experimento, titulo):
    """Agrupa por configuracao: media +- desvio padrao entre sementes."""
    grupos = defaultdict(list)
    for r in _resultados:
        if r["experimento"] == experimento:
            grupos[r["config"]].append(r)

    print(f"\n=== {titulo} ===")
    print(f"{'configuracao':<36} {'n':>2} {'F1 macro':>16} {'acuracia':>16} {'balanceada':>16}")
    linhas = []
    for config, rs in grupos.items():
        f1 = np.array([r["f1_macro"] for r in rs])
        ac = np.array([r["acuracia"] for r in rs])
        ba = np.array([r["balanceada"] for r in rs])
        linhas.append((f1.mean(), config, len(rs), f1, ac, ba))
    for _, config, n, f1, ac, ba in sorted(linhas, key=lambda x: -x[0]):
        print(f"{config:<36} {n:>2} {f1.mean():>8.4f} +-{f1.std():.4f} "
              f"{ac.mean():>8.4f} +-{ac.std():.4f} {ba.mean():>8.4f} +-{ba.std():.4f}", flush=True)


# ---------------------------------------------------------------------------
# experimento 1: phi (Desafio 1)
# ---------------------------------------------------------------------------
VARIANTES_PHI = {
    "sem_phi": (),
    "so_aspecto_sen_cos": ("aspecto",),
    "so_dist_agua": ("agua",),
    "so_elev_x_decl": ("elev_decl",),
    "phi_completo": TERMOS_COMPLETO,
}


def experimento_phi(epocas=20, sementes=(0, 1, 2)):
    """
    O Teorema de Cover diz que levar os dados a uma dimensao maior, com termos
    nao lineares, ajuda um modelo LINEAR a separar classes. Por isso o phi e
    testado em dois modelos: o linear (sem camada oculta), onde o efeito de
    Cover deve aparecer, e a MLP, que aprende suas proprias nao linearidades.
    """
    print("\n##### EXPERIMENTO phi (Desafio 1) #####", flush=True)
    ativ = F.relu
    for tipo, camadas in (("linear", ()), ("mlp", CAMADAS_PADRAO)):
        for nome, termos in VARIANTES_PHI.items():
            for semente in sementes:
                rodar("phi", f"{tipo} / {nome}", termos, camadas, ativ,
                      init_calibrada(ativ, 0.5), epocas, semente)
    resumir("phi", "phi: media +- desvio entre sementes (validacao)")


# ---------------------------------------------------------------------------
# experimento 2: ativacao x inicializacao (Desafio 2)
# ---------------------------------------------------------------------------
def experimento_ativacao(epocas_triagem=12, epocas_final=30, sementes=(0, 1, 2), n_finalistas=5):
    """
    Etapa 1 (triagem): todas as combinacoes, 1 semente, poucas epocas.
    Etapa 2: as melhores, com 3 sementes e mais epocas, para separar efeito
    real de ruido.
    """
    print("\n##### EXPERIMENTO ativacao x inicializacao (Desafio 2) #####", flush=True)
    print("--- Etapa 1: triagem (1 semente) ---", flush=True)
    for nome_a, ativ in ATIVACOES.items():
        for nome_i, fabrica in INICIALIZACOES.items():
            rodar("ativ_triagem", f"{nome_a} + {nome_i}", TERMOS_COMPLETO, CAMADAS_PADRAO,
                  ativ, fabrica(ativ), epocas_triagem, 0)
    resumir("ativ_triagem", "triagem (1 semente, 12 epocas)")

    melhores = defaultdict(list)
    for r in _resultados:
        if r["experimento"] == "ativ_triagem":
            melhores[r["config"]].append(r["f1_macro"])
    finalistas = sorted(melhores, key=lambda c: -np.mean(melhores[c]))[:n_finalistas]

    print(f"\n--- Etapa 2: finalistas ({', '.join(finalistas)}) ---", flush=True)
    for config in finalistas:
        nome_a, nome_i = [parte.strip() for parte in config.split(" + ")]
        ativ = ATIVACOES[nome_a]
        for semente in sementes:
            rodar("ativ_final", config, TERMOS_COMPLETO, CAMADAS_PADRAO, ativ,
                  INICIALIZACOES[nome_i](ativ), epocas_final, semente)
    resumir("ativ_final", "finalistas (3 sementes, 30 epocas)")


# ---------------------------------------------------------------------------
# experimento 3: confirmacao (fecha duas lacunas do experimento 2)
# ---------------------------------------------------------------------------
def experimento_confirmacao(sementes=(0, 1, 2)):
    """
    Lacuna 1: a triagem eliminou todos os ReLU, entao a configuracao que
    estava em uso (relu + calibrada_x0.5) nunca foi comparada com as outras
    em 3 sementes e 30 epocas. Roda aqui, nas mesmas condicoes da etapa 2.

    Lacuna 2: nos 30 epocas todos os finalistas ainda estavam melhorando
    (melhor epoca perto da 30). Compara os principais com 80 epocas, para ver
    se o ranking se mantem com mais treino.
    """
    print("\n##### EXPERIMENTO confirmacao #####", flush=True)

    print("--- (A) ReLU nas mesmas condicoes da etapa 2 (30 epocas) ---", flush=True)
    for config in ("relu + calibrada_x0.5", "relu + calibrada_x1"):
        nome_a, nome_i = [parte.strip() for parte in config.split(" + ")]
        ativ = ATIVACOES[nome_a]
        for semente in sementes:
            rodar("conf_30ep", config, TERMOS_COMPLETO, CAMADAS_PADRAO, ativ,
                  INICIALIZACOES[nome_i](ativ), 30, semente)
    resumir("conf_30ep", "ReLU (3 sementes, 30 epocas)")

    print("\n--- (B) treino mais longo (80 epocas) ---", flush=True)
    for config in ("tanh + lecun", "relu + calibrada_x0.5", "gelu + calibrada_x1"):
        nome_a, nome_i = [parte.strip() for parte in config.split(" + ")]
        ativ = ATIVACOES[nome_a]
        for semente in sementes:
            rodar("conf_80ep", config, TERMOS_COMPLETO, CAMADAS_PADRAO, ativ,
                  INICIALIZACOES[nome_i](ativ), 80, semente)
    resumir("conf_80ep", "treino longo (3 sementes, 80 epocas)")


if __name__ == "__main__":
    pedidos = sys.argv[1:] or ["phi", "ativacao"]
    if "phi" in pedidos:
        experimento_phi()
    if "ativacao" in pedidos:
        experimento_ativacao()
    if "confirmacao" in pedidos:
        experimento_confirmacao()
