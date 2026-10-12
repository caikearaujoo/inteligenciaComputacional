"""
painel.py - gera painel.pdf do Desafio 3 a partir de logs/ e experimentos.csv

Pagina 1: os sinais alinhados (perda, norma do gradiente, norma dos parametros,
          razao de atualizacao, lr efetivo) de 4 execucoes: baseline, melhor
          (Muon), melhor sem Muon (AdamW) e uma instavel (SGD lr 3,0).
          Mesmo eixo de passos em todas: ler a mesma coluna vertical.
Pagina 2: F1 de validacao x lr, por otimizador (etapa 2).
Pagina 3: razao de atualizacao por camada (Muon x AdamW x SGD).
Pagina 4: de onde vem o ganho (decomposicao) e o efeito do orcamento.

Cores: cada OTIMIZADOR tem sempre a mesma cor em todas as paginas (paleta
categorica validada com validate_palette.js). Na pagina 3 as cores sao das
CAMADAS e vem de outros slots da paleta, para nao se confundirem com os
otimizadores; ali o estilo do traco e o rotulo direto tambem identificam a camada.

Uso (da pasta desafio3):  py painel.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, NullFormatter

PASTA = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(PASTA, "logs")

SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
GRADE = "#e6e5e1"
COR = {"muon": "#2a78d6", "adamw": "#eb6834", "adam": "#1baf7a",
       "sgd": "#eda100", "sgd_nesterov": "#e87ba4"}
NOME = {"muon": "Muon", "adamw": "AdamW", "adam": "Adam", "sgd": "SGD", "sgd_nesterov": "SGD Nesterov"}
MARCADOR = {"muon": "o", "adamw": "s", "adam": "^", "sgd": "D", "sgd_nesterov": "v"}

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "text.color": TEXTO, "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "axes.edgecolor": GRADE, "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": TEXTO,
})


def ler(nome, tipo):
    return pd.read_csv(os.path.join(LOGS, f"{nome}_{tipo}.csv"))


def virgula(numero, casas=4):
    return f"{numero:.{casas}f}".replace(".", ",")


def eixo_log(a, minimo, maximo, marcas):
    """Eixo y logaritmico com numeros simples (0,5 em vez de 5x10^-1)."""
    a.set_yscale("log")
    a.set_ylim(minimo, maximo)
    a.set_yticks(marcas)
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------------------
# pagina 1: os sinais alinhados
# ---------------------------------------------------------------------------
EXECUCOES = [
    ("Baseline\nAdam lr 1e-3, constante", "baseline_adam_lr0.001_constante_s0", "adam"),
    ("Melhor (Muon)\nlr 0,03, warmup + cosseno", "muon_lr0.03_lradamw0.01_warmup_cosseno_s0_e40", "muon"),
    ("Melhor sem Muon (AdamW)\nlr 0,01, warmup + cosseno", "adamw_lr0.01_warmup_cosseno_s0_e40", "adamw"),
    ("Instavel (SGD)\nlr 3,0, warmup + cosseno", "sgd_lr3_warmup_cosseno_s0_e40", "sgd"),
]
ROTULOS_LINHAS = ["perda\n(linha: treino, circulo: validacao)", "norma do gradiente ‖g‖",
                  "norma dos parametros ‖θ‖", "razao de atualizacao ρ", "lr efetivo η"]


def pagina_sinais(pdf):
    fig, eixos = plt.subplots(5, 4, figsize=(15, 12.5), sharex=True, squeeze=False)
    for j, (titulo, nome, otim) in enumerate(EXECUCOES):
        passos = ler(nome, "passos")
        epocas = ler(nome, "epocas")
        cor = COR[otim]
        x = passos.passo

        a = eixos[0][j]
        a.plot(x, passos.perda_media, color=cor, lw=1.4)
        a.plot(epocas.passo, epocas.perda_val, "o", mfc=SUPERFICIE, mec=cor, ms=4, mew=1.2, ls="none")
        eixo_log(a, 0.1, 6, [0.1, 0.2, 0.5, 1, 2, 5])
        a.set_title(titulo, loc="left")

        a = eixos[1][j]
        a.plot(x, passos.grad, color=cor, lw=1.0)
        eixo_log(a, 0.1, 4, [0.1, 0.2, 0.5, 1, 2])

        a = eixos[2][j]
        a.plot(x, passos.theta, color=cor, lw=1.6)
        eixo_log(a, 15, 1100, [20, 50, 100, 200, 500, 1000])

        a = eixos[3][j]
        a.plot(x, passos.rho.clip(lower=1e-6), color=cor, lw=1.0)
        a.axhline(1e-3, color=TEXTO_2, lw=0.8, ls=(0, (4, 3)))
        eixo_log(a, 1e-5, 0.1, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

        a = eixos[4][j]
        a.plot(x, passos.lr, color=cor, lw=1.6)
        a.set_ylim(0, passos.lr.max() * 1.1)
        a.set_xlabel("passo")
        a.set_xticks([0, 10000, 20000, 30000])
        a.set_xticklabels(["0", "10 mil", "20 mil", "30 mil"])

    for i in range(5):
        eixos[i][0].set_ylabel(ROTULOS_LINHAS[i])
    eixos[3][0].text(0.02, 1.25e-3, "referencia do enunciado: 1e-3", transform=eixos[3][0].get_yaxis_transform(),
                     ha="left", va="bottom", fontsize=8, color=TEXTO_2)

    # a coluna vertical do instavel: o pico da perda (passo 1700)
    j = 3
    for i in range(5):
        eixos[i][j].axvline(1700, color=TEXTO, lw=0.8, ls=(0, (2, 2)))
    eixos[0][j].text(2600, 0.13, "passo 1700: lr 3,0, perda 4,5,\n‖g‖ 1,4, ‖θ‖ 271 e subindo",
                     fontsize=8, color=TEXTO, va="bottom")
    eixos[2][j].text(30000, 560, "‖θ‖ final: 817", fontsize=8, color=TEXTO, ha="right", va="bottom")

    fig.suptitle("Desafio 3 - os sinais do treino alinhados no mesmo eixo de passos "
                 "(40 epocas = 31.800 passos, semente 0)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.957, "Ler cada coluna de cima para baixo (mesmo passo em todos os graficos). "
             "Os eixos de perda, ‖g‖, ‖θ‖ e ρ sao logaritmicos e iguais entre as colunas.",
             fontsize=9, color=TEXTO_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# pagina 2: F1 x lr por otimizador
# ---------------------------------------------------------------------------
def pagina_lr(pdf, d):
    x = d[(d.agenda == "warmup_cosseno") & (d.epocas == 40) & (d.semente == 0)
          & (d.nome.str.startswith("final_") == False)]
    fig, ax = plt.subplots(figsize=(11, 7))
    melhores = []
    for otim in ["sgd", "sgd_nesterov", "adam", "adamw", "muon"]:
        m = x[x.otimizador == otim].sort_values("lr").drop_duplicates(subset="lr")
        ax.plot(m.lr, m.f1_melhor, color=COR[otim], marker=MARCADOR[otim], lw=1.6, ms=6, label=NOME[otim])
        melhor = m.loc[m.f1_melhor.idxmax()]
        ax.plot([melhor.lr], [melhor.f1_melhor], marker=MARCADOR[otim], ms=14, mfc="none",
                mec=TEXTO, mew=1.2, ls="none")
        melhores.append((otim, melhor.lr, melhor.f1_melhor))
    melhores.sort(key=lambda t: -t[2])

    linhas = ["melhor lr de cada otimizador (F1):"]
    for otim, lr, f1 in melhores:
        linhas.append(f"  {NOME[otim]}: lr {lr:g}  ({virgula(f1)})".replace("lr 0.", "lr 0,"))
    ax.text(0.015, 0.985, "\n".join(linhas), transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=TEXTO, linespacing=1.5)

    base = d[d.nome.str.startswith("baseline")].f1_melhor
    ax.axhline(base.mean(), color=TEXTO_2, lw=1.0, ls=(0, (4, 3)))
    ax.text(3.2e-4, base.mean() - 0.004, "baseline (Adam 1e-3, constante, 3 sementes): " + virgula(base.mean()),
            fontsize=8.5, color=TEXTO_2, va="top")
    ax.annotate("SGD com lr 3,0 colapsou\n(‖θ‖ 817, perda 4,5)", xy=(3.0, 0.6106), xytext=(0.12, 0.575),
                fontsize=8.5, color=TEXTO, arrowprops=dict(arrowstyle="-", color=TEXTO, lw=0.6))
    ax.set_xscale("log")
    ax.set_xlabel("lr de partida (no Muon: o das matrizes ocultas; a parte AdamW fica em 0,01)")
    ax.set_ylabel("F1 macro na validacao (melhor epoca)")
    ax.set_ylim(0.55, 0.99)
    ax.legend(loc="lower left", frameon=False, title="otimizador")
    ax.set_title("F1 x lr por otimizador (warmup + cosseno, 40 epocas, 1 semente)", loc="left")
    fig.text(0.01, 0.005, "Circulo grande = melhor lr de cada otimizador (todos no interior da grade, depois de estender "
             "o SGD ate 3,0). 1 semente por ponto: diferencas < ~0,003 nao separam efeito de ruido.",
             fontsize=8, color=TEXTO_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# pagina 3: rho por camada
# ---------------------------------------------------------------------------
def pagina_rho(pdf):
    colunas = [("Muon (lr 0,03)", "muon_lr0.03_lradamw0.01_warmup_cosseno_s0_e40"),
               ("AdamW (lr 0,01)", "adamw_lr0.01_warmup_cosseno_s0_e40"),
               ("SGD (lr 0,3)", "sgd_lr0.3_warmup_cosseno_s0_e40")]
    # cores das CAMADAS (slots 6, 7 e 8 da paleta: nao sao cores de otimizador) + estilo do traco
    camadas = [("camada 1 (oculta 128x57)", "#4a3aa7", "-"),
               ("camada 2 (oculta 64x128)", "#008300", (0, (6, 2))),
               ("camada 3 (saida 7x64)", "#e34948", (0, (1.5, 1.5)))]
    fig, eixos = plt.subplots(1, 3, figsize=(15, 5.8), sharex=True, sharey=True)
    for j, (titulo, nome) in enumerate(colunas):
        p = ler(nome, "passos")
        meio = p[(p.passo >= 2000) & (p.passo <= 20000)]
        rhos = np.array([meio[f"rho/camadas.{i}.weight"].to_numpy() for i in range(3)])
        razao = np.median(rhos.max(axis=0) / rhos.min(axis=0))
        a = eixos[j]
        for i, (rotulo, cor, estilo) in enumerate(camadas):
            suave = p[f"rho/camadas.{i}.weight"].rolling(20, min_periods=1).mean()
            a.plot(p.passo, suave.clip(lower=1e-6), color=cor, lw=1.7, ls=estilo, label=rotulo)
        a.axhline(1e-3, color=TEXTO_2, lw=0.8, ls=(0, (4, 3)))
        eixo_log(a, 1e-5, 0.1, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        a.set_xlabel("passo")
        a.set_title(titulo, loc="left")
        a.text(0.98, 0.97, "razao max/min entre as 3 camadas\n(mediana, passos 2.000-20.000): " + virgula(razao, 2),
               transform=a.transAxes, ha="right", va="top", fontsize=8.5, color=TEXTO)
    eixos[0].set_ylabel("ρ por camada (media movel de 20 registros)")
    eixos[0].legend(loc="lower left", frameon=False, fontsize=8.5)
    eixos[0].text(0.02, 0.30, "Muon: camadas 1 e 2 no Muon;\ncamada 3 e vieses no AdamW",
                  transform=eixos[0].transAxes, ha="left", fontsize=8, color=TEXTO_2)
    fig.suptitle("Razao de atualizacao por camada: o Muon mexe nas camadas de forma mais parecida",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXTO)
    fig.text(0.01, 0.9, "1 semente por otimizador. Linha tracejada = referencia do enunciado (1e-3). A queda no fim vem da agenda: lr -> 0.",
             fontsize=9, color=TEXTO_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# pagina 4: de onde vem o ganho
# ---------------------------------------------------------------------------
def media_desvio(d, filtro):
    m = d[filtro]
    return m.f1_melhor.mean(), m.f1_melhor.std(ddof=0)


def pagina_ganho(pdf, d):
    x = d[(d.epocas == 40) & (d.semente.isin([0, 1, 2])) & (d.nome.str.startswith("final_") == False)]
    passos = [
        ("Baseline\nAdam lr 1e-3,\nconstante", media_desvio(x, (x.otimizador == "adam") & (x.lr == 0.001) & (x.agenda == "constante")), COR["adam"], "^"),
        ("+ lr 1e-2 e agenda\nwarmup + cosseno\n(Adam)", media_desvio(x, (x.otimizador == "adam") & (x.lr == 0.01) & (x.agenda == "warmup_cosseno")), COR["adam"], "^"),
        ("+ weight decay\n(AdamW)", media_desvio(x, (x.otimizador == "adamw") & (x.lr == 0.01) & (x.agenda == "warmup_cosseno")), COR["adamw"], "s"),
        ("+ Muon nas\nmatrizes ocultas", media_desvio(x, (x.otimizador == "muon") & (x.lr == 0.03) & (x.agenda == "warmup_cosseno")), COR["muon"], "o"),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    medias = [p[1][0] for p in passos]
    a1.plot(range(4), medias, color=TEXTO_2, lw=1.0, zorder=1)
    for i, (rotulo, (media, desvio), cor, marc) in enumerate(passos):
        a1.errorbar(i, media, yerr=desvio, color=TEXTO, capsize=5, lw=1.2, zorder=2)
        a1.plot(i, media, marker=marc, ms=13, color=cor, mec=SUPERFICIE, mew=1.5, zorder=3)
        texto = virgula(media)
        if i > 0:
            texto += "\n(+" + virgula(media - medias[i - 1]) + ")"
        a1.text(i, media + 0.004, texto, ha="center", va="bottom", fontsize=9.5, color=TEXTO)
    a1.set_xticks(range(4))
    a1.set_xticklabels([p[0] for p in passos], fontsize=9)
    a1.set_xlim(-0.5, 3.5)
    a1.set_ylim(0.855, 0.93)
    a1.set_ylabel("F1 macro na validacao (media de 3 sementes ± desvio)")
    a1.set_title("De onde vem o ganho de +0,0475 (40 epocas)", loc="left")
    a1.grid(axis="x", visible=False)

    y = d[(d.epocas == 200) & (d.nome.str.startswith("final_") == False)]

    def media_200(filtro):
        v = y[filtro].f1_melhor
        return v.mean(), v.std(ddof=0), len(v)

    b200 = media_200((y.otimizador == "adam") & (y.lr == 0.001) & (y.agenda == "constante"))
    m200 = media_200((y.otimizador == "muon") & (y.lr == 0.03) & (y.agenda == "warmup_cosseno"))
    ref = [("40 epocas\n(3 sementes)", passos[0][1], passos[3][1]),
           (f"200 epocas\n({b200[2]} sementes)", b200[:2], m200[:2])]
    for k, (rotulo, (base, sd_base), (muon, sd_muon)) in enumerate(ref):
        a2.plot([k, k], [base, muon], color=TEXTO_2, lw=1.0, zorder=1)
        a2.errorbar(k, base, yerr=sd_base, color=TEXTO, capsize=4, lw=1.2, zorder=2)
        a2.errorbar(k, muon, yerr=sd_muon, color=TEXTO, capsize=4, lw=1.2, zorder=2)
        a2.plot(k, base, marker="^", ms=13, color=COR["adam"], mec=SUPERFICIE, mew=1.5, ls="none",
                label="Baseline (Adam 1e-3, constante)" if k == 0 else None, zorder=3)
        a2.plot(k, muon, marker="o", ms=13, color=COR["muon"], mec=SUPERFICIE, mew=1.5, ls="none",
                label="Muon (lr 0,03, warmup + cosseno)" if k == 0 else None, zorder=3)
        a2.text(k + 0.07, base, virgula(base) + " ± " + virgula(sd_base), va="center", fontsize=9, color=TEXTO)
        a2.text(k + 0.07, muon, virgula(muon) + " ± " + virgula(sd_muon), va="center", fontsize=9, color=TEXTO)
        razao = (muon - base) / max(sd_base, sd_muon, 1e-9)
        a2.text(k - 0.07, (base + muon) / 2, "+" + virgula(muon - base) + "\n(" + str(int(razao)) + "x o desvio)",
                va="center", ha="right", fontsize=9.5, color=TEXTO)
    a2.set_xticks(range(2))
    a2.set_xticklabels([r[0] for r in ref])
    a2.set_xlim(-0.5, 1.5)
    a2.set_ylim(0.855, 0.93)
    a2.set_title("O ganho encolhe com mais treino", loc="left")
    a2.legend(loc="lower right", frameon=False, fontsize=8.5)
    a2.grid(axis="x", visible=False)
    fig.suptitle("O que explica o ganho sobre a baseline", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=TEXTO)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    d = pd.read_csv(os.path.join(PASTA, "experimentos.csv"))
    caminho = os.path.join(PASTA, "painel.pdf")
    with PdfPages(caminho) as pdf:
        pagina_sinais(pdf)
        pagina_lr(pdf, d)
        pagina_rho(pdf)
        pagina_ganho(pdf, d)
    print("painel salvo em", caminho)


if __name__ == "__main__":
    main()
