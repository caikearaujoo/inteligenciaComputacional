"""
treino.py - Desafio 3 (GBC073): otimizadores e agendas, observados por dentro
Disciplina: Inteligencia Computacional
Autores: Caike e Izidro

Treina a MLP do projeto (57 -> 128 -> 64 -> 7, tanh, inicializacao calibrada;
a mesma de trabalhoFinal/modelo.py) com o otimizador, a taxa de aprendizado e
a agenda escolhidos na linha de comando, e REGISTRA o que acontece por dentro
do treino. Cada execucao gera:

  logs/<nome>_passos.csv  - um registro a cada k passos (perda, norma do
                            gradiente, norma dos parametros, razao de
                            atualizacao, taxa efetiva; global e por camada)
  logs/<nome>_epocas.csv  - uma linha por epoca (metricas da VALIDACAO)
  experimentos.csv        - uma linha por execucao (resumo + comando exato)

REGRA: so treino e validacao sao usados aqui. O teste nem e carregado em
tensor; ele so sera olhado uma vez, no fim, com a receita vencedora.

Uso (da pasta desafio3):
    py -u treino.py --otimizador adam --lr 1e-3 --agenda constante --semente 0
    py -u treino.py --otimizador muon --lr 0.02 --lr_adamw 3e-3 --agenda warmup_cosseno

Decisoes e justificativas: ver desafio3.md.
"""

import argparse
import contextlib
import copy
import csv
import io
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PASTA, "..", "trabalhoFinal"))
from modelo import MLP                            # noqa: E402
from preprocessamento import executar_pipeline    # noqa: E402

N_CLASSES = 7
PASTA_LOGS = os.path.join(PASTA, "logs")
ARQUIVO_EXPERIMENTOS = os.path.join(PASTA, "experimentos.csv")

OTIMIZADORES = ["sgd", "sgd_nesterov", "adam", "adamw", "muon"]
AGENDAS = ["constante", "cosseno", "warmup_cosseno", "wsd"]

# fracao do treino usada no aquecimento (warmup) e, no WSD, onde comeca a queda
FRACAO_WARMUP = 0.05
FRACAO_ESTAVEL_WSD = 0.80


def ler_argumentos():
    p = argparse.ArgumentParser(description="Desafio 3: otimizador x agenda, com registro")
    p.add_argument("--otimizador", choices=OTIMIZADORES, required=True)
    p.add_argument("--lr", type=float, required=True,
                   help="taxa de aprendizado (no Muon: a das matrizes ocultas)")
    p.add_argument("--lr_adamw", type=float, default=1e-3,
                   help="so no Muon: lr da parte AdamW (vieses e camada de saida)")
    p.add_argument("--agenda", choices=AGENDAS, default="constante")
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--lote", type=int, default=512)
    p.add_argument("--wd", type=float, default=0.01,
                   help="weight decay do AdamW e do Muon (sgd e adam usam 0)")
    p.add_argument("--clip", type=float, default=0.0,
                   help="clipping da norma do gradiente (0 = desligado)")
    p.add_argument("--k", type=int, default=25, help="registrar a cada k passos (10 a 50)")
    p.add_argument("--nome", default="", help="nome da execucao (padrao: gerado)")
    p.add_argument("--salvar", action="store_true",
                   help="salva os pesos da melhor epoca em logs/<nome>.pt (para o teste final)")
    return p.parse_args()


def gerar_nome(a):
    nome = f"{a.otimizador}_lr{a.lr:g}"
    if a.otimizador == "muon":
        nome += f"_lradamw{a.lr_adamw:g}"
    nome += f"_{a.agenda}_s{a.semente}_e{a.epocas}"
    return nome


# ---------------------------------------------------------------------------
# dados (so treino e validacao)
# ---------------------------------------------------------------------------
def carregar_tensores():
    with contextlib.redirect_stdout(io.StringIO()):
        X_tr, X_va, _teste, y_tr, y_va, _y_teste, _ = executar_pipeline()
    return (torch.tensor(X_tr.to_numpy(), dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long),
            torch.tensor(X_va.to_numpy(), dtype=torch.float32), torch.tensor(y_va, dtype=torch.long))


# ---------------------------------------------------------------------------
# otimizadores
# ---------------------------------------------------------------------------
def criar_otimizadores(modelo, a):
    """
    Devolve uma LISTA de otimizadores (1 para todos, 2 para o Muon).

    Muon: so para matrizes 2-D das camadas OCULTAS. Os vieses (1-D) e a camada
    de saida ficam no AdamW, como pede o enunciado. Nesta rede isso deixa 2
    matrizes no Muon (128x57 e 64x128) e 4 tensores no AdamW (os 3 vieses e a
    matriz de saida 7x64).
    """
    if a.otimizador == "sgd":
        return [torch.optim.SGD(modelo.parameters(), lr=a.lr, momentum=0.9)]
    if a.otimizador == "sgd_nesterov":
        return [torch.optim.SGD(modelo.parameters(), lr=a.lr, momentum=0.9, nesterov=True)]
    if a.otimizador == "adam":
        return [torch.optim.Adam(modelo.parameters(), lr=a.lr)]
    if a.otimizador == "adamw":
        return [torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=a.wd)]

    # muon
    n_camadas = len(modelo.camadas)
    ocultas = []
    resto = []
    for n, p in modelo.named_parameters():
        indice_camada = int(n.split(".")[1])
        if p.ndim == 2 and indice_camada < n_camadas - 1:
            ocultas.append(p)
        else:
            resto.append(p)
    muon = torch.optim.Muon(ocultas, lr=a.lr, momentum=0.95, nesterov=True,
                            weight_decay=a.wd, ns_steps=5)
    adamw = torch.optim.AdamW(resto, lr=a.lr_adamw, weight_decay=a.wd)
    return [muon, adamw]


# ---------------------------------------------------------------------------
# agendas de taxa de aprendizado
# ---------------------------------------------------------------------------
def criar_fator_agenda(agenda, total_passos):
    """
    Devolve fator(passo): multiplica o lr base. `passo` = passos ja dados (0, 1, ...).
    Todas as agendas sao por PASSO (nao por epoca): com ~800 passos por epoca,
    o aquecimento e a queda ficam suaves.
    """
    aquecimento = max(1, int(FRACAO_WARMUP * total_passos))
    inicio_queda = int(FRACAO_ESTAVEL_WSD * total_passos)

    def fator(passo):
        if agenda == "constante":
            return 1.0
        if agenda == "cosseno":
            return 0.5 * (1.0 + math.cos(math.pi * passo / total_passos))
        if agenda == "warmup_cosseno":
            if passo < aquecimento:
                return (passo + 1) / aquecimento
            progresso = (passo - aquecimento) / (total_passos - aquecimento)
            return 0.5 * (1.0 + math.cos(math.pi * progresso))
        # wsd: warmup, estavel, queda linear ate 0
        if passo < aquecimento:
            return (passo + 1) / aquecimento
        if passo < inicio_queda:
            return 1.0
        return max(0.0, (total_passos - passo) / (total_passos - inicio_queda))

    return fator


# ---------------------------------------------------------------------------
# medicoes
# ---------------------------------------------------------------------------
def norma_total(tensores):
    return torch.sqrt(sum((t ** 2).sum() for t in tensores)).item()


def nomes_das_camadas(modelo):
    """Pesos da primeira, de uma do meio e da ultima camada linear."""
    pesos = [n for n, _ in modelo.named_parameters() if n.endswith("weight")]
    return [pesos[0], pesos[len(pesos) // 2], pesos[-1]]


def validar(modelo, X_va, y_va):
    modelo.eval()
    with torch.no_grad():
        logits = modelo(X_va)
        perda = F.cross_entropy(logits, y_va).item()
        predito = logits.argmax(dim=1).numpy()
    y = y_va.numpy()
    return (perda, accuracy_score(y, predito), balanced_accuracy_score(y, predito),
            f1_score(y, predito, average="macro", zero_division=0))


def salvar_csv(caminho, linhas):
    if len(linhas) == 0:
        return
    with open(caminho, "w", newline="", encoding="utf-8") as arq:
        escritor = csv.DictWriter(arq, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)


# ---------------------------------------------------------------------------
# treino
# ---------------------------------------------------------------------------
def treinar(a):
    X_tr, y_tr, X_va, y_va = carregar_tensores()

    torch.manual_seed(a.semente)
    modelo = MLP(X_tr.shape[1], N_CLASSES)
    otimizadores = criar_otimizadores(modelo, a)

    passos_por_epoca = math.ceil(len(X_tr) / a.lote)
    total_passos = a.epocas * passos_por_epoca
    fator = criar_fator_agenda(a.agenda, total_passos)
    agendas = [torch.optim.lr_scheduler.LambdaLR(o, fator) for o in otimizadores]

    camadas = nomes_das_camadas(modelo)
    parametros = dict(modelo.named_parameters())
    max_norma = a.clip
    if a.clip <= 0:
        max_norma = float("inf")     # clip_grad_norm_ so mede, sem cortar

    gerador = torch.Generator().manual_seed(a.semente)
    log_passos = []
    log_epocas = []
    passo = 0
    divergiu = False
    soma_perda = 0.0
    n_perda = 0
    melhor_f1, melhor_epoca = -1.0, 0
    melhor_estado = None
    inicio = time.time()

    for epoca in range(1, a.epocas + 1):
        modelo.train()
        soma_perda_epoca = 0.0
        n_epoca = 0
        for idx in torch.randperm(len(X_tr), generator=gerador).split(a.lote):
            passo += 1
            for o in otimizadores:
                o.zero_grad()
            perda = F.cross_entropy(modelo(X_tr[idx]), y_tr[idx])
            perda.backward()

            if not math.isfinite(perda.item()):
                divergiu = True
                break

            registrar = (passo == 1) or (passo % a.k == 0)
            if registrar:
                # antes do clipping e do passo do otimizador
                grad_camada = {n: parametros[n].grad.norm().item() for n in camadas}
                theta_camada = {n: parametros[n].detach().norm().item() for n in camadas}
                theta_global = norma_total(list(modelo.parameters()))
                antes = {n: p.detach().clone() for n, p in parametros.items()}

            # devolve a norma global ANTES do corte (e so corta se --clip > 0)
            grad_global = torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norma).item()

            lr_usado = [o.param_groups[0]["lr"] for o in otimizadores]
            for o in otimizadores:
                o.step()
            for s in agendas:
                s.step()

            soma_perda += perda.item()
            n_perda += 1
            soma_perda_epoca += perda.item() * len(idx)
            n_epoca += len(idx)

            if registrar:
                linha = dict(passo=passo, epoca=epoca, perda=perda.item(),
                             perda_media=soma_perda / n_perda,
                             grad=grad_global, theta=theta_global, lr=lr_usado[0])
                if len(lr_usado) > 1:
                    linha["lr_adamw"] = lr_usado[1]
                diferenca = []
                for n, p in parametros.items():
                    diferenca.append(p.detach() - antes[n])
                linha["rho"] = norma_total(diferenca) / norma_total(list(antes.values()))
                for n in camadas:
                    linha[f"grad/{n}"] = grad_camada[n]
                    linha[f"theta/{n}"] = theta_camada[n]
                    linha[f"rho/{n}"] = ((parametros[n].detach() - antes[n]).norm()
                                         / antes[n].norm()).item()
                log_passos.append(linha)
                soma_perda = 0.0
                n_perda = 0

        if divergiu:
            print(f"  DIVERGIU no passo {passo} (epoca {epoca})", flush=True)
            break

        perda_val, acc, bal, f1 = validar(modelo, X_va, y_va)
        log_epocas.append(dict(epoca=epoca, passo=passo,
                               perda_treino=soma_perda_epoca / n_epoca, perda_val=perda_val,
                               acc_val=acc, balanceada_val=bal, f1_val=f1, lr=lr_usado[0]))
        if f1 > melhor_f1:
            melhor_f1, melhor_epoca = f1, epoca
            melhor_estado = copy.deepcopy(modelo.state_dict())
        if epoca % 10 == 0 or epoca == 1:
            print(f"  epoca {epoca:3d}/{a.epocas} | perda treino {soma_perda_epoca / n_epoca:.4f}"
                  f" | val: perda {perda_val:.4f} acc {acc:.4f} F1 {f1:.4f}", flush=True)

    segundos = time.time() - inicio
    return log_passos, log_epocas, melhor_f1, melhor_epoca, divergiu, segundos, melhor_estado


def main():
    a = ler_argumentos()
    nome = a.nome
    if nome == "":
        nome = gerar_nome(a)
    print(f"Execucao: {nome}", flush=True)

    log_passos, log_epocas, melhor_f1, melhor_epoca, divergiu, segundos, melhor_estado = treinar(a)

    os.makedirs(PASTA_LOGS, exist_ok=True)
    salvar_csv(os.path.join(PASTA_LOGS, f"{nome}_passos.csv"), log_passos)
    salvar_csv(os.path.join(PASTA_LOGS, f"{nome}_epocas.csv"), log_epocas)
    if a.salvar and melhor_estado is not None:
        torch.save(melhor_estado, os.path.join(PASTA_LOGS, f"{nome}.pt"))

    f1_final, acc_final, bal_final = float("nan"), float("nan"), float("nan")
    if len(log_epocas) > 0:
        f1_final = log_epocas[-1]["f1_val"]
        acc_final = log_epocas[-1]["acc_val"]
        bal_final = log_epocas[-1]["balanceada_val"]

    # grava so o que foi realmente usado (wd so no adamw e no muon; lr_adamw so no muon)
    wd_usado = 0.0
    if a.otimizador == "adamw" or a.otimizador == "muon":
        wd_usado = a.wd
    lr_adamw_usado = ""
    if a.otimizador == "muon":
        lr_adamw_usado = a.lr_adamw

    linha = dict(nome=nome, otimizador=a.otimizador, lr=a.lr, lr_adamw=lr_adamw_usado,
                 agenda=a.agenda, semente=a.semente, epocas=a.epocas, lote=a.lote,
                 wd=wd_usado, clip=a.clip, f1_melhor=round(melhor_f1, 4), melhor_epoca=melhor_epoca,
                 f1_final=round(f1_final, 4), acc_final=round(acc_final, 4),
                 balanceada_final=round(bal_final, 4), divergiu=divergiu,
                 segundos=round(segundos, 1), comando="py treino.py " + " ".join(sys.argv[1:]))
    novo = not os.path.exists(ARQUIVO_EXPERIMENTOS)
    with open(ARQUIVO_EXPERIMENTOS, "a", newline="", encoding="utf-8") as arq:
        escritor = csv.DictWriter(arq, fieldnames=list(linha.keys()))
        if novo:
            escritor.writeheader()
        escritor.writerow(linha)

    print(f"Resultado: F1 melhor {melhor_f1:.4f} (epoca {melhor_epoca}), F1 final {f1_final:.4f},"
          f" divergiu={divergiu}, {segundos:.0f}s", flush=True)


if __name__ == "__main__":
    main()
