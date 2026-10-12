"""
etapa4_confirmacao.py - Etapa 4 do protocolo do Desafio 3 (automatica)

1. Espera a etapa 3 terminar (ultima execucao: Muon com agenda WSD).
2. Escolhe as 3 melhores combinacoes (otimizador, lr, agenda) pelo F1 de validacao
   da semente 0 (40 epocas) e roda as sementes 1 e 2 de cada uma.
3. Descobre a vencedora (maior media de F1 em 3 sementes) e roda ela e a baseline
   com 200 epocas (complemento, semente 0): responde a limitacao de a rede nao ter
   saturado em 40 epocas. O orcamento de 40 epocas continua sendo o do protocolo.

Tudo e decidido so pela validacao. A selecao fica impressa no log.
"""

import os
import subprocess
import sys
import time

import pandas as pd

PASTA = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(PASTA, "experimentos.csv")
ULTIMA_DA_ETAPA3 = "muon_lr0.03_lradamw0.01_wsd_s0_e40"


def ler():
    return pd.read_csv(CSV)


def rodar(otimizador, lr, lr_adamw, agenda, semente, epocas):
    comando = [sys.executable, "-u", os.path.join(PASTA, "treino.py"),
               "--otimizador", otimizador, "--lr", str(lr), "--agenda", agenda,
               "--semente", str(semente), "--epocas", str(epocas)]
    if otimizador == "muon":
        comando += ["--lr_adamw", str(lr_adamw)]
    print("RODANDO:", " ".join(comando[2:]), flush=True)
    subprocess.run(comando, cwd=PASTA, check=True)


print("Esperando a etapa 3 terminar...", flush=True)
while ULTIMA_DA_ETAPA3 not in ler().nome.values:
    time.sleep(20)

# ---------------- selecao das 3 melhores combinacoes (semente 0, 40 epocas)
d = ler()
base = d[(d.semente == 0) & (d.epocas == 40) & (d.nome.str.startswith("baseline") == False)].copy()
base["lr_adamw"] = base["lr_adamw"].fillna(0.0)
base = base.sort_values("f1_melhor", ascending=False)
top3 = base.head(3)
print("\nAs 3 melhores combinacoes (semente 0, 40 epocas, validacao):", flush=True)
print(top3[["otimizador", "lr", "lr_adamw", "agenda", "f1_melhor"]].to_string(index=False), flush=True)

# ---------------- sementes 1 e 2 de cada uma
for _, r in top3.iterrows():
    for semente in (1, 2):
        rodar(r.otimizador, r.lr, r.lr_adamw, r.agenda, semente, 40)

# ---------------- media em 3 sementes e vencedora
d = ler()
resumo = []
for _, r in top3.iterrows():
    mesmos = d[(d.otimizador == r.otimizador) & (d.lr == r.lr) & (d.agenda == r.agenda)
               & (d.epocas == 40) & (d.semente.isin([0, 1, 2]))]
    if r.otimizador == "muon":
        mesmos = mesmos[mesmos.lr_adamw == r.lr_adamw]
    resumo.append((mesmos.f1_melhor.mean(), mesmos.f1_melhor.std(ddof=0), len(mesmos), r))
resumo.sort(key=lambda x: -x[0])
print("\nConfirmacao (media +- desvio entre 3 sementes, F1 melhor epoca):", flush=True)
for media, desvio, n, r in resumo:
    print(f"  {r.otimizador} lr {r.lr:g} {r.agenda}: {media:.4f} +- {desvio:.4f} (n={n})", flush=True)

vencedora = resumo[0][3]
print(f"\nVencedora (maior media): {vencedora.otimizador} lr {vencedora.lr:g} {vencedora.agenda}", flush=True)

# ---------------- complemento: 200 epocas da vencedora e da baseline (semente 0)
print("\n--- Complemento: 200 epocas, semente 0 ---", flush=True)
rodar("adam", 0.001, 0.0, "constante", 0, 200)
rodar(vencedora.otimizador, vencedora.lr, vencedora.lr_adamw, vencedora.agenda, 0, 200)
print("\nETAPA 4 COMPLETA", flush=True)
