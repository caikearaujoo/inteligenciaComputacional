"""
slide_html.py - gera a apresentacao do Desafio 3 (slide.html e slide.pdf)

Le os logs e o experimentos.csv e escreve um HTML AUTOSSUFICIENTE (sem internet,
sem bibliotecas): os graficos sao desenhados em SVG pelo proprio HTML, a partir
dos dados embutidos. Assim os numeros do slide nunca divergem dos dados.

Estrutura (4 paginas de 16:9, o limite do professor), em ordem de narrativa:
  1. Ponto de partida: por que mudamos de dataset, o problema, a rede e os termos.
  2. O que o desafio pedia (protocolo em 4 etapas) e as nossas escolhas, com o porque.
  3. O que encontramos: receita vencedora, ganho +- desvio, lr, agendas, confirmacao.
  4. Por que funciona: diagnostico lido nos sinais, decomposicao do ganho, limites
     (200 epocas e a execucao instavel).

Uso (da pasta desafio3):
    py slide_html.py            # gera slide.html
    py slide_html.py --pdf      # gera tambem slide.pdf (usa o Edge ou o Chrome)
"""

import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

PASTA = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(PASTA, "logs")

BASELINE = "baseline_adam_lr0.001_constante_s0"
ADAM = "adam_lr0.01_warmup_cosseno_s0_e40"
ADAMW = "adamw_lr0.01_warmup_cosseno_s0_e40"
MUON = "muon_lr0.03_lradamw0.01_warmup_cosseno_s0_e40"
SGD = "sgd_lr0.3_warmup_cosseno_s0_e40"
SGD_INSTAVEL = "sgd_lr3_warmup_cosseno_s0_e40"
AGENDAS = ["constante", "cosseno", "warmup_cosseno", "wsd"]


def ler(nome):
    return pd.read_csv(os.path.join(LOGS, f"{nome}_passos.csv"))


def serie(nome, coluna, n=170):
    """Passos e valores, reduzidos a ~n pontos (o HTML nao precisa de todos)."""
    p = ler(nome)
    passo = max(1, len(p) // n)
    p = p.iloc[::passo]
    return {"x": [int(v) for v in p.passo], "y": [float(f"{v:.5g}") for v in p[coluna].clip(lower=1e-6)]}


def serie_rho_camadas(nome, n=170):
    """rho de cada uma das 3 camadas, em media movel de 20 registros."""
    p = ler(nome)
    passo = max(1, len(p) // n)
    saida = []
    for i in range(3):
        suave = p[f"rho/camadas.{i}.weight"].rolling(20, min_periods=1).mean().clip(lower=1e-6)
        saida.append({"x": [int(v) for v in p.passo.iloc[::passo]],
                      "y": [float(f"{v:.5g}") for v in suave.iloc[::passo]]})
    return saida


def razao_camadas(nome):
    """Mediana de max/min do rho entre as 3 camadas, passos 2.000 a 20.000."""
    p = ler(nome)
    meio = p[(p.passo >= 2000) & (p.passo <= 20000)]
    rhos = np.array([meio[f"rho/camadas.{i}.weight"].to_numpy() for i in range(3)])
    return float(np.median(rhos.max(axis=0) / rhos.min(axis=0)))


def media_desvio(valores):
    v = np.array(valores, dtype=float)
    return float(v.mean()), float(v.std(ddof=0))


def sementes(d, filtro):
    m = d[filtro].sort_values("semente")
    return [float(v) for v in m.f1_melhor]


def variacao_do_passo(nome, camada):
    """Variacao (CV, %) e mediana de ||delta theta||/lr de uma camada (passos com lr > 1e-3)."""
    p = ler(nome)
    p = p[p.lr > 1e-3]
    v = p[f"rho/camadas.{camada}.weight"] * p[f"theta/camadas.{camada}.weight"] / p.lr
    return float(100 * v.std() / v.mean()), float(v.median())


def montar_dados():
    d = pd.read_csv(os.path.join(PASTA, "experimentos.csv"))
    d = d[d.nome.str.startswith("final_") == False]
    d40 = d[(d.epocas == 40) & (d.semente.isin([0, 1, 2]))]
    d200 = d[d.epocas == 200]

    def cfg(o, lr, ag, base=d40):
        f = (base.otimizador == o) & (base.lr == lr) & (base.agenda == ag)
        if o == "muon":
            f = f & (base.lr_adamw == 0.01)
        return f

    sem_base = sementes(d40, cfg("adam", 0.001, "constante"))
    sem_adam = sementes(d40, cfg("adam", 0.01, "warmup_cosseno"))
    sem_adamw = sementes(d40, cfg("adamw", 0.01, "warmup_cosseno"))
    sem_muon = sementes(d40, cfg("muon", 0.03, "warmup_cosseno"))
    assert len(sem_base) == len(sem_adam) == len(sem_adamw) == len(sem_muon) == 3

    teste = open(os.path.join(PASTA, "teste_final.txt"), encoding="utf-8").read()
    achado = {}
    for b in teste.split("=== ")[1:]:
        nome = "muon" if b.startswith("Receita vencedora") else "baseline"
        achado[nome] = {
            "f1": float(re.search(r"F1 macro\s*:\s*([0-9.]+)", b).group(1)),
            "acc": float(re.search(r"Acuracia simples\s*:\s*([0-9.]+)", b).group(1)),
        }

    # ---- 200 epocas (complemento): todas as sementes que existirem
    s200_base = sementes(d200, cfg("adam", 0.001, "constante", d200))
    s200_muon = sementes(d200, cfg("muon", 0.03, "warmup_cosseno", d200))
    seg200_base = float(d200[cfg("adam", 0.001, "constante", d200)].segundos.mean())
    seg200_muon = float(d200[cfg("muon", 0.03, "warmup_cosseno", d200)].segundos.mean())

    # ---- varredura de lr (etapa 2): warmup + cosseno, semente 0, 40 epocas
    varre_base = d[(d.epocas == 40) & (d.semente == 0) & (d.agenda == "warmup_cosseno")
                   & (d.nome.str.startswith("baseline") == False)
                   & ((d.otimizador != "muon") | (d.lr_adamw == 0.01))]
    varredura = {}
    for o in ["sgd", "sgd_nesterov", "adam", "adamw", "muon"]:
        m = varre_base[varre_base.otimizador == o].sort_values("lr")
        varredura[o] = [{"lr": float(r.lr), "f1": float(r.f1_melhor)} for r in m.itertuples()]

    # ---- agendas (etapa 3): AdamW lr 0,01 e Muon lr 0,03, semente 0
    def f1_s0(o, lr, ag):
        m = d[(d.epocas == 40) & (d.semente == 0) & cfg(o, lr, ag, d)]
        return float(m.f1_melhor.iloc[0])

    agendas = {"adamw": {ag: f1_s0("adamw", 0.01, ag) for ag in AGENDAS},
               "muon": {ag: f1_s0("muon", 0.03, ag) for ag in AGENDAS}}

    # ---- confirmacao (etapa 4): configuracoes com 3 sementes, 40 epocas
    candidatas = [
        ("Muon 0,03 · warmup + cosseno", "muon", 0.03, "warmup_cosseno", "--muon", "c"),
        ("Muon 0,03 · cosseno", "muon", 0.03, "cosseno", "--muon", "c"),
        ("Muon 0,03 · WSD", "muon", 0.03, "wsd", "--muon", "c"),
        ("AdamW 0,01 · warmup + cosseno", "adamw", 0.01, "warmup_cosseno", "--adamw", "q"),
        ("AdamW 0,01 · cosseno", "adamw", 0.01, "cosseno", "--adamw", "q"),
        ("Adam 0,01 · warmup + cosseno", "adam", 0.01, "warmup_cosseno", "--adam", "t"),
        ("Baseline: Adam 1e-3, constante", "adam", 0.001, "constante", "--neutro", "t"),
    ]
    confirmacao = []
    for rotulo, o, lr, ag, cor, forma in candidatas:
        v = sementes(d40, cfg(o, lr, ag))
        assert len(v) == 3, rotulo
        mu, sd = media_desvio(v)
        confirmacao.append({"nome": rotulo, "cor": cor, "forma": forma, "vals": v, "mu": mu, "sd": sd})

    # ---- quantas execucoes em cada etapa
    e40 = d[d.epocas == 40]
    contagem = {
        "total": int(len(d)),
        "etapa1": int(e40.nome.str.startswith("baseline").sum()),
        "etapa2": int(len(varre_base)),
        "etapa3": int(len(e40[(e40.semente == 0) & e40.otimizador.isin(["adamw", "muon"])
                              & e40.agenda.isin(["constante", "cosseno", "wsd"])])),
        "etapa4_40": int(len(e40[e40.semente.isin([1, 2])])),
        "etapa4_200": int(len(d200)),
    }

    # ---- tempo de CPU do protocolo (estimativa): mediana por otimizador x execucoes de 40 epocas
    mediana = e40.groupby("otimizador").segundos.median()
    h40 = float(sum(mediana[o] for o in e40.otimizador)) / 3600
    cpu = {"h40": h40, "h200": 5 * h40}

    # ---- sinais do treino
    cv = {}
    for chave, nome in (("Muon", MUON), ("AdamW", ADAMW), ("SGD", SGD)):
        c0, m0 = variacao_do_passo(nome, 0)
        c1, m1 = variacao_do_passo(nome, 1)
        cv[chave] = {"cv": [c0, c1], "mediana": [m0, m1]}

    theta_final = {}
    for chave, prefixo in (("Adam", "adam_lr0.01_warmup_cosseno"), ("AdamW", "adamw_lr0.01_warmup_cosseno")):
        theta_final[chave] = [float(ler(f"{prefixo}_s{s}_e40").theta.iloc[-1]) for s in (0, 1, 2)]

    def perda_treino(nome):
        return float(pd.read_csv(os.path.join(LOGS, f"{nome}_epocas.csv")).perda_treino.iloc[-1])

    p3 = ler(SGD_INSTAVEL)
    ipico = p3.perda_media.idxmax()
    instavel = {
        "perda": serie(SGD_INSTAVEL, "perda_media"),
        "perda_muon": serie(MUON, "perda_media"),
        "pico": float(p3.perda_media.max()), "passo_pico": int(p3.passo[ipico]),
        "theta_pico": float(p3.theta[ipico]), "theta_final": float(p3.theta.iloc[-1]),
        "f1": float(d[d.nome == SGD_INSTAVEL].f1_melhor.iloc[0]),
    }

    return {
        "n_exec": int(len(d)),
        "n_diverg": int(d.divergiu.sum()),
        "contagem": contagem,
        "cpu": cpu,
        "seeds": {"baseline": sem_base, "adam": sem_adam, "adamw": sem_adamw, "muon": sem_muon},
        "mm": {k: media_desvio(v) for k, v in
               (("baseline", sem_base), ("adam", sem_adam), ("adamw", sem_adamw), ("muon", sem_muon))},
        "teste": achado,
        "e200": {"baseline": s200_base, "muon": s200_muon,
                 "mm_baseline": media_desvio(s200_base), "mm_muon": media_desvio(s200_muon),
                 "custo": seg200_muon / seg200_base},
        "varredura": varredura,
        "agendas": agendas,
        "confirmacao": confirmacao,
        "rho": {"baseline": serie(BASELINE, "rho"), "adam": serie(ADAM, "rho")},
        "perda": {"baseline": serie(BASELINE, "perda_media"), "adam": serie(ADAM, "perda_media")},
        "lr": {"baseline": serie(BASELINE, "lr"), "adam": serie(ADAM, "lr")},
        "theta": {"adam": serie(ADAM, "theta"), "adamw": serie(ADAMW, "theta")},
        "theta_final": theta_final,
        "cv": cv,
        "rho_muon": serie_rho_camadas(MUON),
        "razao": {"Muon": razao_camadas(MUON), "AdamW": razao_camadas(ADAMW), "SGD": razao_camadas(SGD)},
        "muon_previsto": [math.sqrt(57) * math.sqrt(128 / 57), math.sqrt(64)],
        "perda_treino": {"baseline": perda_treino(BASELINE), "adam": perda_treino(ADAM)},
        "rho_pico": {"baseline": float(ler(BASELINE).rho.iloc[2:].max()), "adam": float(ler(ADAM).rho.max())},
        "rho_fim_baseline": float(ler(BASELINE).rho.iloc[-1]),
        "instavel": instavel,
    }


MODELO = r"""<!DOCTYPE html>
<html lang="pt-BR" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Desafio 3 - o melhor treino para o nosso problema</title>
<style>
  :root {
    --bg: #fcfcfb; --card: #f4f3f0; --ink: #0b0b0b; --ink2: #52514e; --grid: #e2e1dc; --borda: #d9d8d3;
    --muon: #2a78d6; --adamw: #eb6834; --adam: #1baf7a; --sgd: #eda100; --nest: #d6559a; --neutro: #8a8984;
    --c1: #4a3aa7; --c2: #008300; --c3: #e34948; --ag2: #7a5af8; --ag4: #c24d86;
    --aviso: #f0efec; --foco: #2a78d6; --ok: #1e8e5a;
  }
  :root[data-theme="dark"] {
    --bg: #1a1a19; --card: #232322; --ink: #ffffff; --ink2: #c3c2b7; --grid: #383835; --borda: #3a3a37;
    --muon: #3987e5; --adamw: #d95926; --adam: #199e70; --sgd: #c98500; --nest: #e0679f; --neutro: #8a8984;
    --c1: #8b7fe0; --c2: #3fb23f; --c3: #f0716f; --ag2: #9a85ff; --ag4: #d8689e;
    --aviso: #232322; --foco: #3987e5; --ok: #36b277;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
    font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif; overflow: hidden; }
  #palco { position: absolute; left: 0; top: 0; width: 1600px; height: 900px; transform-origin: 0 0; }
  .pagina { position: absolute; left: 0; top: 0; width: 1600px; height: 900px; display: none; background: var(--bg); overflow: hidden; }
  .pagina.atual { display: block; }
  .cab { position: absolute; left: 40px; top: 16px; width: 1520px; }
  h1 { margin: 0; font-size: 33px; line-height: 1.12; letter-spacing: -.01em; }
  .sub { margin-top: 6px; font-size: 17px; color: var(--ink2); }
  .npag { position: absolute; right: 40px; bottom: 10px; font-size: 14px; color: var(--ink2); font-variant-numeric: tabular-nums; }
  .bloco { position: absolute; background: var(--card); border: 1px solid var(--borda); border-radius: 14px; padding: 14px 18px; }
  h2 { margin: 0 0 8px; font-size: 21px; line-height: 1.2; }
  .rotulo { font-size: 15px; color: var(--ink2); }
  .chips { display: flex; flex-direction: column; gap: 9px; margin-top: 4px; }
  .chip { display: flex; align-items: center; gap: 10px; font-size: 17px; line-height: 1.28; }
  .ponto { width: 14px; height: 14px; border-radius: 50%; flex: none; }
  .ponto.q { border-radius: 3px; }
  .grande { font-size: 54px; font-weight: 700; color: var(--muon); letter-spacing: -.02em; line-height: 1; font-variant-numeric: tabular-nums; }
  .medio { font-size: 38px; font-weight: 700; color: var(--muon); letter-spacing: -.01em; line-height: 1; font-variant-numeric: tabular-nums; }
  .num { font-variant-numeric: tabular-nums; }
  .tag { display: inline-block; font-size: 15px; font-weight: 700; padding: 3px 10px; border-radius: 999px;
    background: var(--bg); border: 1px solid var(--borda); margin-left: 8px; vertical-align: 2px; }
  .texto { font-size: 17px; line-height: 1.42; margin: 8px 0 0; }
  .texto b { font-weight: 700; }
  .pequeno { font-size: 15px; line-height: 1.4; color: var(--ink2); }
  .lido { font-size: 14.5px; color: var(--ink2); margin-top: 8px; }
  ul.lista { margin: 8px 0 0; padding-left: 22px; font-size: 17px; line-height: 1.45; }
  ul.lista li { margin-bottom: 6px; }
  ul.lista.curta { font-size: 16px; }
  .ok { color: var(--ok); font-weight: 700; }
  .etapa { font-size: 15px; font-weight: 700; color: var(--ink2); letter-spacing: .04em; text-transform: uppercase; }
  .linha { height: 1px; background: var(--borda); margin: 12px 0; }
  table.t { border-collapse: collapse; width: 100%; font-size: 16px; font-variant-numeric: tabular-nums; margin-top: 6px; }
  table.t th, table.t td { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--grid); }
  table.t th { font-size: 14px; color: var(--ink2); font-weight: 600; }
  svg { display: block; overflow: visible; }
  svg text { font-family: inherit; }
  .eixo { stroke: var(--grid); stroke-width: 1; }
  .grade { stroke: var(--grid); stroke-width: 1; }
  .tick { fill: var(--ink2); font-size: 13px; }
  .rot { font-size: 14px; font-weight: 600; }
  #dica { position: absolute; pointer-events: none; z-index: 20; background: var(--ink); color: var(--bg); font-size: 14px;
    padding: 7px 10px; border-radius: 8px; line-height: 1.4; opacity: 0; transition: opacity .1s; white-space: nowrap; font-variant-numeric: tabular-nums; }
  /* paginas 1 e 2: componentes novos */
  .cadeia { display: flex; align-items: stretch; gap: 6px; margin-top: 4px; }
  .cadeia .passo { flex: 1; background: var(--bg); border: 1px solid var(--borda); border-radius: 10px; padding: 8px 10px; font-size: 13.5px; line-height: 1.34; }
  .cadeia .passo .et { font-size: 13px; font-weight: 700; color: var(--ink2); margin-bottom: 3px; }
  .cadeia .passo.destaque { border-color: var(--muon); }
  .cadeia.v { flex-direction: column; gap: 3px; }
  .cadeia.v .passo { font-size: 14px; line-height: 1.36; padding: 6px 10px; }
  .cadeia.v .passo .et { display: inline; margin: 0 6px 0 0; font-size: 14px; color: var(--ink); }
  .cadeia.v .passo .et::after { content: ":"; }
  .cadeia.v .seta { align-self: flex-start; font-size: 14px; line-height: .9; margin-left: 16px; }
  .cadeia .seta { align-self: center; font-size: 20px; color: var(--ink2); }
  ul.legenda { list-style: none; margin: 6px 0 0; padding: 0; font-size: 16px; line-height: 1.3; }
  ul.legenda li { display: flex; align-items: center; gap: 10px; margin-bottom: 13px; }
  ul.legenda .desc { font-size: 13px; color: var(--ink2); }
  .mini { background: var(--bg); border: 1px solid var(--borda); border-radius: 10px; padding: 10px 12px; }
  /* interface (some na impressao) */
  .ui { position: fixed; z-index: 50; }
  #barra { right: 14px; bottom: 6px; display: flex; gap: 6px; align-items: center; opacity: .45; transition: opacity .2s; }
  #barra:hover { opacity: 1; }
  #barra button { font: inherit; font-size: 13px; padding: 5px 10px; border-radius: 8px; border: 1px solid var(--borda);
    background: var(--card); color: var(--ink); cursor: pointer; }
  #barra button:hover { border-color: var(--foco); }
  #relogio { font-variant-numeric: tabular-nums; font-size: 22px; font-weight: 700; padding: 2px 12px; border-radius: 8px;
    background: var(--ink); color: var(--bg); display: none; }
  #relogio.fim { background: #d03b3b; color: #fff; }
  #gaveta { right: 0; top: 0; bottom: 0; width: 460px; background: var(--card); border-left: 1px solid var(--borda);
    padding: 54px 18px 18px; overflow: auto; display: none; font-size: 15px; line-height: 1.5; }
  #gaveta h3 { margin: 16px 0 6px; font-size: 16px; }
  #gaveta table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; margin: 4px 0 8px; }
  #gaveta th, #gaveta td { text-align: left; padding: 3px 8px 3px 0; border-bottom: 1px solid var(--grid); }
  #gaveta .abas { display: flex; gap: 6px; margin-bottom: 10px; }
  #gaveta .abas button { font: inherit; padding: 4px 10px; border-radius: 8px; border: 1px solid var(--borda); background: var(--bg); color: var(--ink); cursor: pointer; }
  #gaveta .abas button.sel { border-color: var(--foco); font-weight: 700; }
  #ajuda { left: 14px; bottom: 6px; font-size: 12.5px; color: var(--ink2); display: none; background: var(--card); padding: 4px 8px; border-radius: 8px; border: 1px solid var(--borda); }
  @page { size: 1600px 900px; margin: 0; }
  @media print {
    .ui, #dica { display: none !important; }
    html, body { overflow: visible; width: 1600px; height: auto; background: #fcfcfb; }
    #palco { position: static; transform: none !important; width: 1600px; height: auto; }
    .pagina { display: block !important; position: relative; width: 1600px; height: 900px;
      page-break-after: always; break-after: page; }
    .pagina:last-child { page-break-after: auto; break-after: auto; }
  }
</style>
</head>
<body>
<div id="palco">
  <div id="dica"></div>

  <!-- ===================== 1 · PONTO DE PARTIDA ===================== -->
  <section class="pagina atual" id="p1">
    <div class="cab"><h1>1 · Ponto de partida: por que o Covertype e o que precisamos treinar</h1>
      <div class="sub">Trocamos de dataset para ter um problema com espaço para aplicar a disciplina; o Desafio 3 pergunta como treinar bem a rede que esse problema comporta</div></div>

    <div class="bloco" style="left:40px; top:104px; width:800px; height:436px;">
      <h2>Por que mudamos de dataset</h2>
      <table class="t" style="font-size:16px">
        <tr><th></th><th>Dataset inicial</th><th>Forest Covertype (atual)</th></tr>
        <tr><td>Objetivo</td><td>Recomendar a cultura a plantar, a partir das condições de solo e clima</td><td><b>Prever a cobertura vegetal</b> de uma célula de terreno, a partir de relevo, solo e distâncias</td></tr>
        <tr><td>Linhas</td><td>2.200</td><td><b>581.012</b></td></tr>
        <tr><td>Classes</td><td>22 culturas</td><td><b>7 coberturas vegetais</b></td></tr>
        <tr><td>Acerto no teste</td><td>98,6% (chute: 4,5%)</td><td>91,3% na baseline (chute: 48,8%)</td></tr>
        <tr><td>Situação</td><td>Saturado: já acertava quase tudo</td><td>Há margem para melhorar</td></tr>
      </table>
      <div class="texto" style="font-size:16px; margin-top:12px"><b>O problema do dataset inicial:</b> era pequeno e fácil demais. Com 98,6% de acerto, qualquer mudança no treino seria invisível, e uma rede maior não cabia na regra de ~20 exemplos de treino por parâmetro.</div>
      <div class="texto" style="font-size:16px; margin-top:8px"><b>O que o Covertype dá:</b> volume para uma rede de tamanho razoável dentro dessa regra, classes desbalanceadas como no mundo real e margem para evoluir. Queríamos dar mais <b>espaço para aplicar os conceitos da disciplina</b> (inicialização, ativação, otimizadores).</div>
    </div>

    <div class="bloco" style="left:860px; top:104px; width:700px; height:436px;">
      <h2>O problema e a rede que vamos treinar</h2>
      <div class="texto" style="font-size:15.5px; margin-top:2px">Cada exemplo é uma célula de terreno de 30×30 m de uma floresta dos EUA (Colorado), descrita por <b>elevação, distâncias até a água e a focos de incêndio, sombreamento, tipo de solo</b> e outras variáveis. <b>Objetivo da rede:</b> dizer qual das <b>7 coberturas vegetais</b> domina ali.</div>
      <svg id="svgRede" width="664" height="176" style="margin-top:4px"></svg>
      <ul class="lista curta" id="lRede" style="font-size:14px; margin-top:2px; line-height:1.36"></ul>
    </div>

    <div class="bloco" style="left:40px; top:552px; width:520px; height:326px;">
      <h2>Como a rede aprende</h2>
      <div class="texto" style="font-size:14.5px; margin-top:0">Treinar é <b>ajustar esses 16 mil parâmetros para o erro cair</b>:</div>
      <div class="cadeia v">
        <div class="passo"><div class="et">1 · Função de perda</div>Mede o quanto a rede errou. Transforma os 7 números em probabilidades: é <b>alta</b> quando a rede dá pouca probabilidade à cobertura certa e <b>baixa</b> quando dá muita.</div>
        <div class="seta">↓</div>
        <div class="passo"><div class="et">2 · Gradiente</div>Para cada peso, para onde a perda cresce.</div>
        <div class="seta">↓</div>
        <div class="passo destaque"><div class="et">3 · Otimizador</div>A regra que usa o gradiente para atualizar os pesos a cada passo. O que muda de um para outro é o <b>tamanho e a direção do passo</b>: é o que o Desafio 3 compara.</div>
      </div>
    </div>

    <div class="bloco" style="left:580px; top:552px; width:480px; height:326px;">
      <h2>Ativação: trocamos o ReLU pelo tanh <span class="pequeno" style="font-weight:400">(Desafio 2)</span></h2>
      <svg id="svgAtiv" width="444" height="112"></svg>
      <ul class="lista curta" id="lAtiv" style="font-size:14px; margin-top:4px; line-height:1.34"></ul>
    </div>

    <div class="bloco" style="left:1080px; top:552px; width:480px; height:326px;">
      <h2>Baseline: o ponto de comparação</h2>
      <div class="texto" style="font-size:15px; margin-top:2px">O enunciado pede para manter a <b>receita atual do grupo</b>, mas ainda não tínhamos uma, porque tínhamos acabado de trocar de dataset. Então definimos o treino padrão: <b>Adam, lr 1e-3, agenda constante</b>. Essa é a nossa <b>baseline</b>.</div>
      <div class="linha"></div>
      <div class="texto" style="font-size:15px; margin-top:0">Durante o desenvolvimento, comparamos tudo com ela e buscamos uma <b>receita melhor</b> (otimizador + lr + agenda): é o que o Desafio 3 pede.</div>
      <div class="pequeno" style="margin-top:12px; font-size:14px">A pergunta é se existe uma receita melhor e, se existir, <b>por que</b> ela é melhor.</div>
    </div>
    <div class="npag">1 / 4</div>
  </section>

  <!-- ===================== 2 · O QUE O DESAFIO PEDIA ===================== -->
  <section class="pagina" id="p2">
    <div class="cab"><h1>2 · O que o desafio pedia e como respondemos</h1>
      <div class="sub" id="sub2"></div></div>

    <div class="bloco" style="left:40px; top:100px; width:610px; height:284px;">
      <h2>O que o enunciado pede para comparar</h2>
      <div style="display:grid; grid-template-columns:1.3fr 1fr; gap:0 18px; margin-top:2px">
        <div><div class="etapa">Otimizadores</div><ul class="legenda" id="lOtim"></ul></div>
        <div><div class="etapa">Agendas de lr</div><ul class="legenda" id="lAg"></ul></div>
      </div>
    </div>

    <div class="bloco" style="left:666px; top:100px; width:894px; height:284px;">
      <h2>Agendas: como o lr muda ao longo do treino <span class="pequeno" style="font-weight:400">(fração do lr máximo)</span></h2>
      <div style="display:flex; gap:22px; align-items:flex-start">
        <svg id="svgCurvas" width="470" height="214" style="flex:none"></svg>
        <div class="texto" style="font-size:15.5px; margin-top:0" id="tWsd"></div>
      </div>
    </div>

    <div class="bloco" style="left:40px; top:398px; width:610px; height:248px;">
      <h2>Por que 40 épocas em tudo (e 200 só para confirmar)</h2>
      <div id="tempoCpu"></div>
      <div class="texto" style="font-size:15px; margin-top:8px" id="tEpocas"></div>
    </div>

    <div class="bloco" style="left:666px; top:398px; width:894px; height:248px;">
      <h2>O protocolo do enunciado, em 4 etapas</h2>
      <div id="etapas" style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-top:4px"></div>
    </div>

    <div class="bloco" style="left:40px; top:658px; width:1520px; height:206px;">
      <h2>Regras do enunciado e como cumprimos</h2>
      <div id="regras" style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px 14px; margin-top:4px"></div>
      <div class="pequeno" style="margin-top:12px; font-size:14px"><b>Outras decisões nossas</b> (registro a cada 25 passos, só o lr ajustado, sementes extras de AdamW e Adam): estão nas notas (tecla N).</div>
    </div>
    <div class="npag">2 / 4</div>
  </section>

  <!-- ===================== 3 · RESULTADO ===================== -->
  <section class="pagina" id="p3">
    <div class="cab"><h1 id="t3"></h1>
      <div class="sub">Do melhor lr de cada otimizador, passando pelas agendas, até a confirmação com 3 sementes</div></div>

    <div class="bloco" style="left:40px; top:104px; width:470px; height:756px;">
      <h2>Receita vencedora</h2>
      <div class="chips" style="gap:7px">
        <div class="chip" style="font-size:16px"><span class="ponto" style="background:var(--muon)"></span><span><b>Muon</b>, lr 0,03, nas 2 matrizes ocultas (momento 0,95, Nesterov)</span></div>
        <div class="chip" style="font-size:16px"><span class="ponto q" style="background:var(--adamw)"></span><span><b>AdamW</b>, lr 0,01, nos vieses e na camada de saída</span></div>
        <div class="chip" style="font-size:16px"><span class="ponto" style="background:var(--ink2); border-radius:2px; height:5px; width:18px"></span><span>agenda <b>warmup (5%) + cosseno até 0</b></span></div>
        <div class="chip" style="font-size:16px"><span class="ponto" style="background:transparent; border:2px solid var(--ink2)"></span><span>40 épocas = 31.800 passos, lote 512, igual para todos</span></div>
      </div>
      <div class="linha"></div>
      <h2>Ganho (F1 macro, validação)</h2>
      <div class="etapa">40 épocas · 3 sementes</div>
      <div class="grande" id="g40" style="margin-top:6px"></div>
      <div class="rotulo" style="margin-top:6px; font-size:15.5px" id="g40b"></div>
      <svg id="svgSem40" width="434" height="92" style="margin-top:6px"></svg>
      <div class="texto" id="g40r" style="margin-top:2px; font-size:16px"></div>
      <div class="linha"></div>
      <div class="texto" id="gTeste" style="margin-top:0; font-size:16px"></div>
      <div class="pequeno" style="margin-top:8px">É a receita adotada no modelo final do projeto (<code>modelo.py</code>).</div>
      <div class="texto" style="font-size:15px; margin-top:8px">As duas nuvens de pontos não se tocam: a diferença não é sorte de semente.</div>
    </div>

    <div class="bloco" style="left:530px; top:104px; width:520px; height:756px;">
      <h2>Etapa 2 · o melhor lr de cada otimizador</h2>
      <svg id="svgVar" width="484" height="296"></svg>
      <ul class="lista curta" id="lVar" style="font-size:15px; margin-top:6px"></ul>
      <div class="linha"></div>
      <h2>Etapa 3 · as agendas (F1, 1 semente)</h2>
      <svg id="svgAgDots" width="484" height="130"></svg>
      <div class="texto" id="tAg" style="font-size:15px; margin-top:4px"></div>
    </div>

    <div class="bloco" style="left:1070px; top:104px; width:490px; height:756px;">
      <h2>Etapa 4 · confirmação (3 sementes)</h2>
      <svg id="svgConf" width="454" height="380"></svg>
      <ul class="lista curta" id="lConf" style="font-size:15px; margin-top:6px"></ul>
    </div>
    <div class="npag">3 / 4</div>
  </section>

  <!-- ===================== 4 · POR QUE FUNCIONA E ONDE PARA ===================== -->
  <section class="pagina" id="p4">
    <div class="cab"><h1>4 · Por que funciona: o que os sinais mostram e onde o ganho para de valer</h1>
      <div class="sub">O diagnóstico lido no treino, a decomposição do ganho e as limitações, inclusive a execução que falhou</div></div>

    <div class="bloco" style="left:40px; top:104px; width:480px; height:756px;">
      <h2>De onde vem o ganho</h2>
      <svg id="svgDec" width="444" height="440"></svg>
      <div class="texto" style="font-size:15px; margin-top:4px"><b style="color:var(--adam)" id="tg1"></b> · <span id="r1"></span></div>
      <div class="texto" style="font-size:15px; margin-top:6px"><b style="color:var(--adamw)" id="tg2"></b> · <span id="r2"></span></div>
      <div class="texto" style="font-size:15px; margin-top:6px"><b style="color:var(--muon)" id="tg3"></b> · <span id="r3"></span></div>
    </div>

    <div class="bloco" style="left:540px; top:104px; width:500px; height:756px;">
      <h2>Diagnóstico: lr maior + queda a zero</h2>
      <div class="rotulo">ρ = ‖Δθ‖/‖θ‖ (tamanho do passo, escala log)</div>
      <svg id="svgP1rho" width="464" height="250"></svg>
      <div class="rotulo" style="margin-top:6px">perda de treino (escala log)</div>
      <svg id="svgD1b" width="464" height="220"></svg>
      <ul class="lista curta" id="lD1" style="font-size:15px; margin-top:6px"></ul>
    </div>

    <div class="bloco" style="left:1060px; top:104px; width:500px; height:756px;">
      <h2>Limites do resultado</h2>
      <div class="rotulo">Com mais treino o ganho encolhe (1 ponto = 1 semente)</div>
      <svg id="svgLim" width="464" height="190"></svg>
      <div class="texto" id="tLim" style="font-size:15px; margin-top:4px"></div>
      <div class="linha"></div>
      <div class="rotulo">Execução instável: SGD com lr 3,0 (perda de treino)</div>
      <svg id="svgInst" width="464" height="210"></svg>
      <div class="texto" id="tInst" style="font-size:15px; margin-top:4px"></div>
    </div>
    <div class="npag">4 / 4</div>
  </section>
</div>

<div id="barra" class="ui">
  <button id="bAnt" title="←">◀</button>
  <span id="pos" style="font-size:13px">1 / 4</span>
  <button id="bProx" title="→">▶</button>
  <span id="relogio">3:00</span>
  <button id="bRel" title="T">⏱ Cronômetro</button>
  <button id="bNotas" title="N">📝 Notas</button>
  <button id="bTema" title="D">◐ Tema</button>
  <button id="bPdf" title="P">⬇ PDF</button>
</div>
<div id="gaveta" class="ui">
  <div class="abas"><button id="aNotas" class="sel">Notas do orador</button><button id="aDados">Dados</button></div>
  <div id="painelNotas"></div>
  <div id="painelDados" style="display:none"></div>
</div>
<div id="ajuda" class="ui">← → páginas · 1 a 4 vai à página · T cronômetro (espaço pausa, R zera) · N notas · V dados · D tema · F tela cheia · P imprimir/PDF · ? esta ajuda</div>

<script>
const D = /*__DADOS__*/;
const $ = id => document.getElementById(id);

/* ---------- utilidades ---------- */
const NS = "http://www.w3.org/2000/svg";
const cor = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, c = 4) => v.toFixed(c).replace(".", ",");
const svgEl = (tag, attrs, pai) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (pai) pai.appendChild(e);
  return e;
};
const texto = (svg, x, y, s, attrs = {}) => { const t = svgEl("text", { x, y, ...attrs }, svg); t.textContent = s; return t; };
let escala = 1;
const dica = $("dica");
function mostrarDica(html, ev) {
  const palco = $("palco").getBoundingClientRect();
  dica.innerHTML = html;
  let x = (ev.clientX - palco.left) / escala + 14, y = (ev.clientY - palco.top) / escala - 8;
  if (x > 1380) x -= 280;
  dica.style.left = x + "px"; dica.style.top = y + "px"; dica.style.opacity = 1;
}
const esconderDica = () => { dica.style.opacity = 0; };
const OPT = { sgd: ["SGD", "--sgd", "d"], sgd_nesterov: ["SGD Nesterov", "--nest", "v"], adam: ["Adam", "--adam", "t"],
              adamw: ["AdamW", "--adamw", "q"], muon: ["Muon", "--muon", "c"] };
const razaoTxt = (v, n) => n >= 2 ? Math.floor(v) + "×" : "—";

function marca(svg, forma, x, y, r, c) {
  const at = { fill: c, stroke: cor("--card"), "stroke-width": 2 };
  if (forma === "q") return svgEl("rect", { x: x - r, y: y - r, width: 2 * r, height: 2 * r, rx: 2, ...at }, svg);
  if (forma === "t") return svgEl("path", { d: `M${x} ${y - r - 2}L${x + r + 1} ${y + r - 1}L${x - r - 1} ${y + r - 1}Z`, ...at }, svg);
  if (forma === "v") return svgEl("path", { d: `M${x} ${y + r + 2}L${x + r + 1} ${y - r + 1}L${x - r - 1} ${y - r + 1}Z`, ...at }, svg);
  if (forma === "d") return svgEl("path", { d: `M${x} ${y - r - 2}L${x + r + 2} ${y}L${x} ${y + r + 2}L${x - r - 2} ${y}Z`, ...at }, svg);
  return svgEl("circle", { cx: x, cy: y, r, ...at }, svg);
}

/* ---------- grafico de linhas (eixos x linear, y linear ou log) ---------- */
function graficoLinhas(svg, o) {
  svg.innerHTML = "";
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height");
  const m = { l: 56, r: 12, t: 10, b: o.semX ? 8 : 30 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const fy = o.log ? Math.log10 : (v => v);
  const sx = v => m.l + (v - o.x0) / (o.x1 - o.x0) * iw;
  const sy = v => m.t + ih - (fy(v) - fy(o.y0)) / (fy(o.y1) - fy(o.y0)) * ih;
  o.yticks.forEach(t => {
    svgEl("line", { x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t), class: "grade" }, svg);
    texto(svg, m.l - 8, sy(t) + 4, o.yfmt(t), { "text-anchor": "end", class: "tick" });
  });
  if (!o.semX) o.xticks.forEach(t => texto(svg, sx(t), H - 8, t === 0 ? "0" : (t / 1000) + " mil", { "text-anchor": "middle", class: "tick" }));
  svgEl("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, class: "eixo" }, svg);
  (o.ref || []).forEach(r => {
    svgEl("line", { x1: m.l, x2: m.l + iw, y1: sy(r.y), y2: sy(r.y), stroke: cor("--ink2"), "stroke-width": 1, "stroke-dasharray": "5 4" }, svg);
    texto(svg, m.l + iw, sy(r.y) - 5, r.texto, { "text-anchor": "end", class: "tick" });
  });
  o.series.forEach(s => {
    let d = "";
    s.x.forEach((x, i) => { d += (i ? "L" : "M") + sx(x).toFixed(1) + " " + sy(Math.max(s.y[i], o.y0)).toFixed(1); });
    svgEl("path", { d, fill: "none", stroke: cor(s.cor), "stroke-width": 2.3, "stroke-linejoin": "round" }, svg);
  });
  (o.rotulos || []).forEach(r => {
    const tx = svgEl("text", { x: sx(r.x), y: sy(r.y), "text-anchor": r.ancora || "start", class: "rot", fill: cor(r.cor) }, svg);
    String(r.texto).split("\n").forEach((linha, i) => { const ts = svgEl("tspan", { x: sx(r.x), dy: i ? 16 : 0 }, tx); ts.textContent = linha; });
  });
  const guia = svgEl("line", { y1: m.t, y2: m.t + ih, stroke: cor("--ink"), "stroke-width": 1, opacity: 0 }, svg);
  const area = svgEl("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
  area.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / escala;
    const passo = o.x0 + (px - m.l) / iw * (o.x1 - o.x0);
    let html = "<b>passo " + Math.round(passo).toLocaleString("pt-BR") + "</b>";
    o.series.forEach(s => {
      let k = 0, melhor = 1e18;
      s.x.forEach((x, i) => { const dd = Math.abs(x - passo); if (dd < melhor) { melhor = dd; k = i; } });
      html += "<br><span style='color:" + cor(s.cor) + "'>●</span> " + s.nome + ": " + o.tip(s.y[k]);
    });
    guia.setAttribute("x1", px); guia.setAttribute("x2", px); guia.setAttribute("opacity", .5);
    mostrarDica(html, ev);
  });
  area.addEventListener("mouseleave", () => { guia.setAttribute("opacity", 0); esconderDica(); });
}

/* ---------- pontos por semente (media +- desvio) ---------- */
function sementesGen(svg, o) {
  svg.innerHTML = "";
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height");
  const m = { l: o.ml, r: o.mr, t: o.cap ? 24 : 8, b: 26 };
  const sx = v => m.l + (v - o.lo) / (o.hi - o.lo) * (W - m.l - m.r);
  const rowH = (H - m.t - m.b) / o.linhas.length;
  o.ticks.forEach(t => {
    svgEl("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b, class: "grade" }, svg);
    texto(svg, sx(t), H - m.b + 17, fmt(t, 2), { "text-anchor": "middle", class: "tick" });
  });
  if (o.cap) texto(svg, m.l, 12, o.cap, { class: "tick" });
  svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "eixo" }, svg);
  o.linhas.forEach((l, i) => {
    const y = m.t + (i + 0.5) * rowH;
    if (l.destaque) svgEl("rect", { x: 0, y: y - rowH / 2 + 2, width: W, height: rowH - 4, rx: 8, fill: cor("--bg"), opacity: .9 }, svg);
    texto(svg, 0, y + 5, l.nome, { class: "rot", fill: cor("--ink"), "font-weight": l.destaque ? 700 : 600 });
    svgEl("line", { x1: sx(l.mu - l.sd), x2: sx(l.mu + l.sd), y1: y, y2: y, stroke: cor(l.cor), "stroke-width": 7, opacity: .35, "stroke-linecap": "round" }, svg);
    l.vals.forEach((v, k) => {
      const p = marca(svg, l.forma || "c", sx(v), y, 7, cor(l.cor));
      p.addEventListener("mousemove", ev => mostrarDica("<b>" + l.nome + "</b><br>semente " + (l.sem ? l.sem[k] : k) + ": F1 " + fmt(v), ev));
      p.addEventListener("mouseleave", esconderDica);
    });
    if (o.valores) texto(svg, W - 2, y + 5, fmt(l.mu) + " ± " + fmt(l.sd), { "text-anchor": "end", class: "rot", fill: cor("--ink") });
  });
}

/* ---------- pagina 1: a rede ---------- */
function desenharRede() {
  const svg = $("svgRede"); svg.innerHTML = "";
  const cols = [[57, "57 entradas", 70], [128, "128 neurônios", 250], [64, "64 neurônios", 430], [7, "7 classes", 610]];
  const alt = v => 18 + Math.sqrt(v) * 9.5, cy = 88;
  const ligacoes = [["128×57", "Muon", "--muon"], ["64×128", "Muon", "--muon"], ["7×64", "AdamW", "--adamw"]];
  ligacoes.forEach(([mat, quem, c], i) => {
    const x1 = cols[i][2] + 20, x2 = cols[i + 1][2] - 20;
    svgEl("line", { x1, x2, y1: cy, y2: cy, stroke: cor(c), "stroke-width": 4 }, svg);
    svgEl("path", { d: `M${x2} ${cy} l-9 -6 l0 12 Z`, fill: cor(c) }, svg);
    texto(svg, (x1 + x2) / 2, cy - 14, mat, { "text-anchor": "middle", class: "rot", fill: cor("--ink") });
    texto(svg, (x1 + x2) / 2, cy + 22, quem, { "text-anchor": "middle", class: "rot", fill: cor(c) });
  });
  cols.forEach(([v, rotulo, x], i) => {
    const h = alt(v);
    svgEl("rect", { x: x - 20, y: cy - h / 2, width: 40, height: h, rx: 8, fill: cor("--card"), stroke: cor("--ink2"), "stroke-width": 2 }, svg);
    texto(svg, x, cy + 5, String(v), { "text-anchor": "middle", class: "rot", fill: cor("--ink"), "font-size": 16 });
    texto(svg, x, 168, rotulo, { "text-anchor": "middle", class: "tick", fill: cor("--ink") });
  });
  texto(svg, 0, 12, "cada seta = uma matriz de pesos (e o otimizador que a atualiza)", { class: "tick" });
}

/* ---------- pagina 1: ReLU x tanh ---------- */
function desenharAtivacao() {
  const svg = $("svgAtiv"); svg.innerHTML = "";
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height"), m = { l: 30, r: 8, t: 6, b: 20 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b, x0 = -3, x1 = 3, y0 = -1.4, y1 = 3;
  const sx = v => m.l + (v - x0) / (x1 - x0) * iw, sy = v => m.t + ih * (1 - (v - y0) / (y1 - y0));
  [-1, 0, 1, 2, 3].forEach(v => {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "grade" }, svg);
    texto(svg, m.l - 6, sy(v) + 4, String(v).replace("-", "−"), { "text-anchor": "end", class: "tick" });
  });
  [-3, -2, -1, 0, 1, 2, 3].forEach(v => texto(svg, sx(v), H - 4, String(v).replace("-", "−"), { "text-anchor": "middle", class: "tick" }));
  svgEl("line", { x1: sx(0), x2: sx(0), y1: m.t, y2: H - m.b, stroke: cor("--ink2"), "stroke-width": 1 }, svg);
  [-1, 1].forEach(v => svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: cor("--ag2"), "stroke-width": 1, "stroke-dasharray": "4 4", opacity: .8 }, svg));
  const curva = (fn, c, w) => {
    let d = "";
    for (let i = 0; i <= 120; i++) { const x = x0 + (x1 - x0) * i / 120; d += (i ? "L" : "M") + sx(x).toFixed(1) + " " + sy(fn(x)).toFixed(1); }
    svgEl("path", { d, fill: "none", stroke: cor(c), "stroke-width": w, "stroke-linejoin": "round" }, svg);
  };
  curva(x => Math.max(0, x), "--neutro", 2.6);
  curva(x => Math.tanh(x), "--ag2", 3);
  texto(svg, sx(-2.9), sy(2.55), "ReLU: zera os", { class: "rot", fill: cor("--ink2") });
  texto(svg, sx(-2.9), sy(1.75), "negativos, sem teto", { class: "rot", fill: cor("--ink2") });
  texto(svg, sx(0.3), sy(-0.82), "tanh: limitada entre −1 e 1", { class: "rot", fill: cor("--ag2") });
}

/* ---------- pagina 2: legendas e curvas das agendas ---------- */
const AGEST = { constante: ["constante", "--neutro", "5 4", 2], cosseno: ["cosseno", "--ag2", "", 2.6], warmup_cosseno: ["warmup + cosseno", "--c2", "", 2.6], wsd: ["WSD (nossa escolha)", "--ag4", "", 3.4] };
function desenharLegendas() {
  const icone = (forma, c) => { const sv = svgEl("svg", { width: 18, height: 18 }); marca(sv, forma, 9, 9, 6, cor(c)); return sv.outerHTML; };
  const desc = { sgd: "passo ∝ gradiente", sgd_nesterov: "", adam: "passo normalizado", adamw: "Adam + weight decay", muon: "passo ortogonalizado" };
  const nomes = { sgd: "SGD", sgd_nesterov: "SGD Nesterov", adam: "Adam", adamw: "AdamW", muon: "Muon (obrigatório)" };
  $("lOtim").innerHTML = ["sgd", "sgd_nesterov", "adam", "adamw", "muon"].map(k =>
    "<li>" + icone(OPT[k][2], OPT[k][1]) + "<span><b>" + nomes[k] + "</b>" + (desc[k] ? " <span class='desc'>" + desc[k] + "</span>" : "") + "</span></li>").join("");
  $("lAg").innerHTML = AGENDAS.map(a => {
    const [nome, c, dash, w] = AGEST[a];
    const sv = svgEl("svg", { width: 26, height: 14 });
    svgEl("line", { x1: 1, x2: 25, y1: 7, y2: 7, stroke: cor(c), "stroke-width": w, "stroke-dasharray": dash }, sv);
    return "<li>" + sv.outerHTML + "<span><b>" + nome + "</b></span></li>";
  }).join("");
}
function desenharCurvasAgenda() {
  const svg = $("svgCurvas"); svg.innerHTML = "";
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height"), m = { l: 36, r: 8, t: 10, b: 36 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const sx = t => m.l + t * iw, sy = v => m.t + ih * (1 - v);
  const aq = 0.05, est = 0.8;
  const f = {
    constante: t => 1,
    cosseno: t => 0.5 * (1 + Math.cos(Math.PI * t)),
    warmup_cosseno: t => t < aq ? t / aq : 0.5 * (1 + Math.cos(Math.PI * (t - aq) / (1 - aq))),
    wsd: t => t < aq ? t / aq : (t < est ? 1 : Math.max(0, (1 - t) / (1 - est)))
  };
  [0, 0.5, 1].forEach(v => {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "grade" }, svg);
    texto(svg, m.l - 6, sy(v) + 4, String(v).replace(".", ","), { "text-anchor": "end", class: "tick" });
  });
  [0, 0.25, 0.5, 0.75, 1].forEach(t => texto(svg, sx(t), H - 18, Math.round(t * 100) + "%", { "text-anchor": "middle", class: "tick" }));
  texto(svg, m.l + iw / 2, H - 2, "tempo de treino", { "text-anchor": "middle", class: "tick" });
  ["constante", "cosseno", "warmup_cosseno", "wsd"].forEach(a => {
    const [nome, c, dash, w] = AGEST[a];
    let d = "";
    for (let i = 0; i <= 200; i++) { const t = i / 200; d += (i ? "L" : "M") + sx(t).toFixed(1) + " " + sy(f[a](t)).toFixed(1); }
    svgEl("path", { d, fill: "none", stroke: cor(c), "stroke-width": w, "stroke-dasharray": dash, "stroke-linejoin": "round" }, svg);
  });
  texto(svg, sx(0.55), sy(0.74), "WSD: alto até 80%", { class: "rot", fill: cor("--ag4") });
  texto(svg, sx(0.04), sy(0.34), "cosseno: desce desde o início", { class: "rot", fill: cor("--ag2") });
  const guia = svgEl("line", { y1: m.t, y2: m.t + ih, stroke: cor("--ink"), "stroke-width": 1, opacity: 0 }, svg);
  const area = svgEl("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
  area.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / escala, t = Math.max(0, Math.min(1, (px - m.l) / iw));
    let html = "<b>" + Math.round(t * 100) + "% do treino</b>";
    ["constante", "cosseno", "warmup_cosseno", "wsd"].forEach(a => { html += "<br><span style='color:" + cor(AGEST[a][1]) + "'>●</span> " + AGEST[a][0] + ": " + fmt(f[a](t), 2); });
    guia.setAttribute("x1", px); guia.setAttribute("x2", px); guia.setAttribute("opacity", .5);
    mostrarDica(html, ev);
  });
  area.addEventListener("mouseleave", () => { guia.setAttribute("opacity", 0); esconderDica(); });
}

/* ---------- pagina 3: ganho por semente (40 epocas) ---------- */
const media = v => v.reduce((a, b) => a + b, 0) / v.length;
const desvio = v => { const m = media(v); return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); };
const linhaSem = (nome, c, forma, vals) => ({ nome, cor: c, forma, vals, mu: media(vals), sd: desvio(vals) });
function desenharSementes() {
  sementesGen($("svgSem40"), { ml: 76, mr: 20, lo: 0.855, hi: 0.93, ticks: [0.86, 0.88, 0.90, 0.92], cap: "cada ponto = 1 semente (F1 de validação)",
    linhas: [linhaSem("Muon", "--muon", "c", D.seeds.muon), linhaSem("Baseline", "--neutro", "t", D.seeds.baseline)] });
}

/* ---------- pagina 3: varredura de lr ---------- */
function desenharVarredura() {
  const svg = $("svgVar"); svg.innerHTML = "";
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height"), m = { l: 46, r: 14, t: 34, b: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const lx0 = Math.log10(2e-4), lx1 = Math.log10(6), y0 = 0.55, y1 = 0.95;
  const sx = v => m.l + (Math.log10(v) - lx0) / (lx1 - lx0) * iw;
  const sy = v => m.t + ih * (1 - (v - y0) / (y1 - y0));
  for (let v = 0.55; v <= 0.951; v += 0.1) {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "grade" }, svg);
    texto(svg, m.l - 6, sy(v) + 4, fmt(v, 2), { "text-anchor": "end", class: "tick" });
  }
  [[1e-3, "1e-3"], [1e-2, "1e-2"], [1e-1, "1e-1"], [1, "1"]].forEach(([v, s]) => {
    svgEl("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "grade" }, svg);
    texto(svg, sx(v), H - m.b + 17, s, { "text-anchor": "middle", class: "tick" });
  });
  texto(svg, m.l + iw / 2, H - 6, "lr de partida (escala log)", { "text-anchor": "middle", class: "tick" });
  svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "eixo" }, svg);
  const bm = D.mm.baseline[0];
  svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(bm), y2: sy(bm), stroke: cor("--ink2"), "stroke-width": 1.4, "stroke-dasharray": "6 5" }, svg);
  texto(svg, m.l + 4, sy(bm) - 6, "baseline: " + fmt(bm), { class: "tick" });
  Object.entries(D.varredura).forEach(([k, pts]) => {
    const [nome, c, forma] = OPT[k];
    let d = "";
    pts.forEach((p, i) => { d += (i ? "L" : "M") + sx(p.lr).toFixed(1) + " " + sy(p.f1).toFixed(1); });
    svgEl("path", { d, fill: "none", stroke: cor(c), "stroke-width": 2.2, opacity: .9 }, svg);
    let melhor = pts[0];
    pts.forEach(p => {
      if (p.f1 > melhor.f1) melhor = p;
      const mk = marca(svg, forma, sx(p.lr), sy(p.f1), 5, cor(c));
      mk.addEventListener("mousemove", ev => mostrarDica("<b>" + nome + "</b><br>lr " + String(p.lr).replace(".", ",") + "<br>F1 " + fmt(p.f1), ev));
      mk.addEventListener("mouseleave", esconderDica);
    });
    svgEl("circle", { cx: sx(melhor.lr), cy: sy(melhor.f1), r: 11, fill: "none", stroke: cor("--ink"), "stroke-width": 1.8 }, svg);
  });
  const dx = [0, 58, 172, 232, 302];
  Object.entries(OPT).forEach(([k, [nome, c, forma]], i) => {
    marca(svg, forma, m.l + dx[i] + 5, 11, 5, cor(c));
    texto(svg, m.l + dx[i] + 15, 15, nome, { class: "tick", fill: cor("--ink") });
  });
  texto(svg, m.l + 366, 15, "◯ = melhor lr", { class: "tick" });
  const pi = D.varredura.sgd.find(p => p.lr === 3);
  if (pi) texto(svg, sx(3) - 10, sy(pi.f1) + 26, "SGD lr 3,0 colapsou", { "text-anchor": "end", class: "tick", fill: cor("--sgd"), "font-weight": 700 });
}

/* ---------- pagina 3: agendas ---------- */
const AGENDAS = ["constante", "cosseno", "warmup_cosseno", "wsd"];
const AGNOME = { constante: "constante", cosseno: "cosseno", warmup_cosseno: "warmup + cosseno", wsd: "WSD" };
function desenharAgendas() {
  const svg = $("svgAgDots"); svg.innerHTML = "";
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height"), m = { l: 124, r: 120, t: 20, b: 24 }, lo = 0.80, hi = 0.92;
  const sx = v => m.l + (v - lo) / (hi - lo) * (W - m.l - m.r);
  const rowH = (H - m.t - m.b) / 4;
  [0.80, 0.84, 0.88, 0.92].forEach(t => {
    svgEl("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b, class: "grade" }, svg);
    texto(svg, sx(t), H - m.b + 16, fmt(t, 2), { "text-anchor": "middle", class: "tick" });
  });
  texto(svg, m.l, 11, "■ AdamW lr 0,01     ● Muon lr 0,03", { class: "tick" });
  AGENDAS.forEach((ag, i) => {
    const y = m.t + (i + 0.5) * rowH, a = D.agendas.adamw[ag], mu = D.agendas.muon[ag];
    texto(svg, 0, y + 5, AGNOME[ag], { class: "rot", fill: cor("--ink") });
    svgEl("line", { x1: sx(a), x2: sx(mu), y1: y, y2: y, stroke: cor("--ink2"), "stroke-width": 1.4 }, svg);
    marca(svg, "q", sx(a), y, 6, cor("--adamw")); marca(svg, "c", sx(mu), y, 6, cor("--muon"));
    texto(svg, W, y + 5, fmt(a, 3) + "  ·  " + fmt(mu, 3), { "text-anchor": "end", class: "rot", fill: cor("--ink") });
  });
}

/* ---------- pagina 3: confirmacao (3 sementes) ---------- */
function desenharConfirmacao() {
  sementesGen($("svgConf"), { ml: 222, mr: 112, lo: 0.855, hi: 0.925, ticks: [0.86, 0.88, 0.90, 0.92], cap: "cada ponto = 1 semente (F1 de validação)", valores: true,
    linhas: D.confirmacao.map((c, i) => ({ ...c, destaque: i === 0 })) });
}

/* ---------- pagina 4: decomposicao do ganho ---------- */
function desenharDecomposicao() {
  const svg = $("svgDec"); svg.innerHTML = "";
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height"), m = { l: 50, r: 10, t: 34, b: 54 }, lo = 0.855, hi = 0.93;
  const sx = i => m.l + (i + 0.5) / 4 * (W - m.l - m.r);
  const sy = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
  [0.86, 0.88, 0.90, 0.92].forEach(t => {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "grade" }, svg);
    texto(svg, m.l - 6, sy(t) + 4, fmt(t, 2), { "text-anchor": "end", class: "tick" });
  });
  const passos = [
    ["Baseline", "Adam 1e-3", "baseline", "--neutro", "t"],
    ["+ lr e agenda", "Adam 1e-2", "adam", "--adam", "t"],
    ["+ weight decay", "AdamW", "adamw", "--adamw", "q"],
    ["+ Muon", "Muon+AdamW", "muon", "--muon", "c"],
  ];
  let d = "";
  passos.forEach((p, i) => { d += (i ? "L" : "M") + sx(i) + " " + sy(D.mm[p[2]][0]); });
  svgEl("path", { d, fill: "none", stroke: cor("--ink2"), "stroke-width": 1.6 }, svg);
  passos.forEach((p, i) => {
    const [mu, sd] = D.mm[p[2]], x = sx(i), y = sy(mu);
    svgEl("line", { x1: x, x2: x, y1: sy(mu - sd), y2: sy(mu + sd), stroke: cor("--ink"), "stroke-width": 2 }, svg);
    D.seeds[p[2]].forEach(v => svgEl("circle", { cx: x + 17, cy: sy(v), r: 3, fill: cor(p[3]), opacity: .55 }, svg));
    const mk = marca(svg, p[4], x, y, 9, cor(p[3]));
    texto(svg, x, y - 18, fmt(mu), { "text-anchor": "middle", class: "rot", fill: cor("--ink"), "font-size": 15 });
    if (i) texto(svg, x + 14, y + 28, "+" + fmt(mu - D.mm[passos[i - 1][2]][0]), { "text-anchor": "start", class: "tick", fill: cor("--ink2"), "font-size": 14 });
    texto(svg, x, H - 30, p[0], { "text-anchor": "middle", class: "rot", fill: cor("--ink"), "font-size": 14 });
    texto(svg, x, H - 12, p[1], { "text-anchor": "middle", class: "tick" });
    mk.addEventListener("mousemove", ev => mostrarDica("<b>" + p[1] + "</b><br>F1 " + fmt(mu) + " ± " + fmt(sd) + "<br>sementes: " + D.seeds[p[2]].map(v => fmt(v)).join(" · "), ev));
    mk.addEventListener("mouseleave", esconderDica);
  });
  texto(svg, m.l, 14, "pontos pequenos = as 3 sementes · barra = ± desvio", { class: "tick" });
}

/* ---------- pagina 4: diagnostico (rho e perda) ---------- */
function desenharRho(id) {
  graficoLinhas($(id), {
    x0: 0, x1: 31800, y0: 1e-5, y1: 0.05, log: true, yticks: [1e-5, 1e-4, 1e-3, 1e-2], xticks: [0, 10000, 20000, 30000],
    yfmt: v => v.toExponential(0), tip: v => "ρ = " + v.toExponential(1),
    ref: [{ y: 1e-3, texto: "referência 1e-3" }],
    series: [{ nome: "Baseline (Adam 1e-3, constante)", cor: "--neutro", x: D.rho.baseline.x, y: D.rho.baseline.y },
             { nome: "Adam lr 1e-2 + warmup/cosseno", cor: "--adam", x: D.rho.adam.x, y: D.rho.adam.y }],
    rotulos: [{ x: 5000, y: 1.8e-4, texto: "baseline: ρ ~1e-3 → 5e-4", cor: "--ink2" },
              { x: 9000, y: 1.35e-2, texto: "lr 1e-2: pico 6e-3, depois → 0", cor: "--adam" }]
  });
}
function desenharD1() {
  desenharRho("svgP1rho");
  graficoLinhas($("svgD1b"), {
    x0: 0, x1: 31800, y0: 0.1, y1: 2.5, log: true, yticks: [0.1, 0.2, 0.5, 1, 2], xticks: [0, 10000, 20000, 30000],
    yfmt: v => String(v).replace(".", ","), tip: v => "perda = " + fmt(v, 3),
    series: [{ nome: "Baseline", cor: "--neutro", x: D.perda.baseline.x, y: D.perda.baseline.y },
             { nome: "Adam lr 1e-2", cor: "--adam", x: D.perda.adam.x, y: D.perda.adam.y }],
    rotulos: [{ x: 31300, y: D.perda_treino.baseline * 1.4, texto: "baseline: " + fmt(D.perda_treino.baseline, 3), cor: "--ink2", ancora: "end" },
              { x: 31300, y: D.perda_treino.adam * 0.7, texto: "lr 1e-2: " + fmt(D.perda_treino.adam, 3), cor: "--adam", ancora: "end" }]
  });
}

/* ---------- pagina 4: limitacoes ---------- */
function desenharLimitacoes() {
  sementesGen($("svgLim"), { ml: 128, mr: 120, lo: 0.855, hi: 0.935, ticks: [0.86, 0.88, 0.90, 0.92], valores: true,
    linhas: [linhaSem("Muon · 40 épocas", "--muon", "c", D.seeds.muon), linhaSem("Baseline · 40 ép.", "--neutro", "t", D.seeds.baseline),
             linhaSem("Muon · 200 épocas", "--muon", "c", D.e200.muon), linhaSem("Baseline · 200 ép.", "--neutro", "t", D.e200.baseline)] });
  graficoLinhas($("svgInst"), {
    x0: 0, x1: 31800, y0: 0.1, y1: 6, log: true, yticks: [0.1, 0.5, 1, 5], xticks: [0, 10000, 20000, 30000],
    yfmt: v => String(v).replace(".", ","), tip: v => "perda = " + fmt(v, 3),
    series: [{ nome: "SGD lr 3,0", cor: "--sgd", x: D.instavel.perda.x, y: D.instavel.perda.y },
             { nome: "Muon (a vencedora)", cor: "--muon", x: D.instavel.perda_muon.x, y: D.instavel.perda_muon.y }],
    rotulos: [{ x: D.instavel.passo_pico + 1500, y: D.instavel.pico * 0.9, texto: "SGD lr 3,0: pico " + fmt(D.instavel.pico, 1), cor: "--sgd" },
              { x: 14000, y: 0.22, texto: "Muon (vencedora)", cor: "--muon" }]
  });
}

/* ---------- textos que dependem dos dados ---------- */
function preencherTextos() {
  const [mm, ms] = D.mm.muon, [bm, bs] = D.mm.baseline, c = D.contagem;
  const [m2, s2] = D.e200.mm_muon, [b2, bs2] = D.e200.mm_baseline, n2 = D.e200.muon.length;
  const pct = v => (v * 100).toFixed(1).replace(".", ",") + "%";
  const exp = v => v.toExponential(0).replace("-", "‑");
  const vezes = (a, b) => Math.floor(a / b);
  const melhor = k => D.varredura[k].reduce((a, b) => b.f1 > a.f1 ? b : a);
  const lrTxt = v => String(v).replace(".", ",");
  const nSem = n2 === 1 ? "1 semente" : n2 + " sementes";

  /* pagina 1 */
  $("lRede").innerHTML = [
    "<b>Cada camada oculta</b> multiplica pelos pesos, soma o viés e aplica a <b>tanh</b>. A camada de saída só multiplica e soma, e entrega um <b>vetor de 7 números (logits)</b>, um por cobertura: a <b>posição do maior número</b> é a cobertura prevista.",
    "<b>16.135 parâmetros</b> (pesos e vieses de cada camada): cerca de <b>25 exemplos de treino por parâmetro</b> (406.708). Os 57 valores de entrada vêm das 54 variáveis originais, com as contínuas padronizadas."].map(s => "<li>" + s + "</li>").join("");
  $("lAtiv").innerHTML = [
    "Testamos as ativações com <b>3 sementes cada</b>, na validação. A ideia de testar o tanh veio das <b>apresentações dos colegas</b> (muita gente o usou).",
    "O tanh foi a <b>melhor</b>: 3 desvios ou mais acima do GELU e do ReLU.",
    "Ele <b>limita o sinal entre −1 e 1</b>, o que ajuda a manter o treino estável."].map(s => "<li>" + s + "</li>").join("");

  /* pagina 2 */
  $("sub2").textContent = "O enunciado define o protocolo em 4 etapas; o que ficou aberto nós decidimos e declaramos · " + c.total + " execuções reportadas, " + (D.n_diverg === 0 ? "nenhuma divergiu" : D.n_diverg + " divergiram");
  const cartao = (et, titulo, corpo, res) => "<div class='mini'><div class='etapa'>" + et + "</div><div style='font-size:16.5px; font-weight:700; line-height:1.2; margin:3px 0 6px'>" + titulo +
    "</div><div class='texto' style='font-size:13.5px; margin-top:0; line-height:1.36'>" + corpo + "</div><div class='linha' style='margin:8px 0'></div><div class='texto' style='font-size:13.5px; margin-top:0; line-height:1.36'>" + res + "</div></div>";
  $("etapas").innerHTML = [
    cartao("Etapa 1", "Baseline", "A receita do grupo: Adam, lr 1e-3, constante. 3 sementes.", "<b>" + c.etapa1 + " execuções</b><br>F1 " + fmt(bm) + " ± " + fmt(bs)),
    cartao("Etapa 2", "Varredura de lr", "5 a 7 lrs por otimizador (warmup + cosseno, 1 semente). O SGD caiu na borda: estendemos.", "<b>" + c.etapa2 + " execuções</b><br>melhor: Muon, lr " + lrTxt(melhor("muon").lr)),
    cartao("Etapa 3", "Agendas", "As 4 agendas no AdamW e no Muon, cada um no seu melhor lr.", "<b>" + c.etapa3 + " execuções</b><br>vence warmup + cosseno"),
    cartao("Etapa 4", "Confirmação", "As melhores, com 3 sementes, mais AdamW e Adam como controle e 200 épocas.", "<b>" + (c.etapa4_40 + c.etapa4_200) + " execuções</b><br>" + c.etapa4_40 + " de 40 e " + c.etapa4_200 + " de 200 épocas")].join("");
  $("regras").innerHTML = [
    "<b>lr ajustado</b> por otimizador (Muon sobre o melhor lr do AdamW)",
    "<b>Mesmo orçamento</b> (passos e lote) em todas as comparações",
    "<b>Validação decide</b>; teste usado <b>uma única vez</b>",
    "<b>Tudo reportado</b>: " + (D.n_diverg === 0 ? "nenhuma divergiu; SGD lr 3,0 colapsou (F1 " + fmt(D.instavel.f1, 2) + ")" : D.n_diverg + " divergiram"),
    "<b>Vencedor só se o ganho supera o desvio</b>: +" + fmt(mm - bm, 3) + " contra " + fmt(ms, 4) + " e " + fmt(bs, 4),
    "<b>Muon incluído</b> (<code>torch.optim.Muon</code>, só nas matrizes ocultas)",
    "<b>Reprodutível</b>: cada linha do <code>experimentos.csv</code> traz o comando exato",
    "<b>Agenda à escolha</b>: WSD, que contrasta com o cosseno"].map(t => "<div class='mini' style='padding:9px 12px; font-size:14.5px; line-height:1.32'><span class='ok'>✓</span> " + t + "</div>").join("");
  const custo = D.e200.custo.toFixed(1).replace(".", ",");
  const h40 = D.cpu.h40, h200 = D.cpu.h200, larg = 270, horas = v => v.toFixed(1).replace(".", ",");
  $("tempoCpu").innerHTML = "<div style='display:grid; grid-template-columns:172px 1fr; gap:8px 10px; align-items:center; margin-top:2px'>" +
    "<div style='font-size:15px'><b>40 épocas</b> em tudo<div class='pequeno' style='font-size:13px'>o que usamos</div></div>" +
    "<div style='display:flex; align-items:center; gap:8px'><div style='height:22px; width:" + Math.round(larg * h40 / h200) + "px; background:var(--ok); border-radius:4px'></div><b class='num' style='font-size:17px'>≈ " + horas(h40) + " h</b></div>" +
    "<div style='font-size:15px'><b>200 épocas</b> em tudo<div class='pequeno' style='font-size:13px'>estimativa: 5× o tempo</div></div>" +
    "<div style='display:flex; align-items:center; gap:8px'><div style='height:22px; width:" + larg + "px; background:var(--neutro); border-radius:4px'></div><b class='num' style='font-size:17px'>≈ " + horas(h200) + " h</b></div></div>" +
    "<div class='pequeno' style='margin-top:6px; font-size:13px'>Tempo de CPU do protocolo inteiro (sem GPU).</div>";
  $("tEpocas").innerHTML = "O Muon é o mais lento (~" + custo + "× por época). Usamos <b>40 épocas, igual para todos</b>; depois <b>confirmamos com 200 épocas só a vencedora e a baseline</b>, porque ficava a dúvida se a vantagem do Muon era só aprender mais rápido.";
  $("tWsd").innerHTML = "<b>Cosseno:</b> começa a baixar o lr desde o início, suave, até 0.<br><br>" +
    "<b>WSD (nossa escolha):</b> sobe (warmup), <b>mantém o lr alto por 80%</b> do treino e só cai no final.<br><br>" +
    "Escolhemos o WSD porque está nas leituras do enunciado e contrasta com o cosseno. Ele não ganhou, mas mostrou que o que importa é a <b>queda do lr até perto de 0 no fim</b>.";

  /* pagina 3 */
  $("t3").textContent = "3 · O que encontramos: Muon + queda do lr dá +" + fmt(mm - bm, 3) + " de F1 sobre a baseline";
  $("g40").textContent = fmt(mm) + " ± " + fmt(ms);
  $("g40b").innerHTML = "contra <b class='num'>" + fmt(bm) + " ± " + fmt(bs) + "</b> da baseline (Adam 1e-3, constante)";
  $("g40r").innerHTML = "<b>+" + fmt(mm - bm) + "</b> = " + vezes(mm - bm, Math.max(ms, bs)) + "× o desvio entre sementes";
  $("gTeste").innerHTML = "<b>Teste (usado uma única vez):</b> F1 <b class='num'>" + fmt(D.teste.muon.f1) + "</b> contra " + fmt(D.teste.baseline.f1) +
    " · acurácia " + pct(D.teste.muon.acc) + " contra " + pct(D.teste.baseline.acc);
  const ordem = ["muon", "adamw", "adam", "sgd_nesterov", "sgd"];
  $("lVar").innerHTML = [
    "<b>Melhor lr:</b> " + ordem.map(k => OPT[k][0] + " " + lrTxt(melhor(k).lr) + " (" + fmt(melhor(k).f1, 3) + ")").join(" · ") + ". Todos <b>no interior</b> da grade.",
    "O SGD com lr 3,0 <b>colapsou</b> (F1 " + fmt(D.instavel.f1, 2) + "): é a execução instável (página 4). 1 semente por ponto: diferenças menores que ~0,003 não separam efeito de ruído."].map(s => "<li>" + s + "</li>").join("");
  $("tAg").innerHTML = "Com lr <b>constante</b> o Muon perde do AdamW (<b>" + fmt(D.agendas.muon.constante, 3) + "</b> contra <b>" + fmt(D.agendas.adamw.constante, 3) +
    "</b>): a vantagem do Muon exige a queda do lr. O WSD ficou abaixo do warmup + cosseno nos dois.";
  const cf = D.confirmacao, mu1 = cf[0], mu2 = cf[1], mu3 = cf[2];
  $("lConf").innerHTML = [
    "<b>Vencedora:</b> Muon lr 0,03 + warmup/cosseno, <b>" + fmt(mu1.mu) + " ± " + fmt(mu1.sd) + "</b> contra " + fmt(cf[6].mu) + " ± " + fmt(cf[6].sd) + " da baseline: <b>" + vezes(mm - bm, Math.max(ms, bs)) + "× o desvio</b>, então há vencedor.",
    "Cosseno puro empata (" + fmt(mu2.mu) + " ± " + fmt(mu2.sd) + "): a diferença é menor que o desvio, então o <b>warmup não fez diferença</b> aqui. O WSD fica " + fmt(mu1.mu - mu3.mu, 4) + " abaixo.",
    "AdamW (" + fmt(cf[3].mu) + ") contra Adam (" + fmt(cf[5].mu) + ") no mesmo lr e agenda: o weight decay soma +" + fmt(cf[3].mu - cf[5].mu, 4) + "."].map(s => "<li>" + s + "</li>").join("");

  /* pagina 4 */
  const g1 = D.mm.adam[0] - bm, g2 = D.mm.adamw[0] - D.mm.adam[0], g3 = mm - D.mm.adamw[0];
  $("tg1").textContent = "1 · lr + agenda: +" + fmt(g1, 3) + " (⅔ do ganho)";
  $("tg2").textContent = "2 · weight decay: +" + fmt(g2, 3);
  $("tg3").textContent = "3 · Muon: +" + fmt(g3, 3) + " (¼ do ganho)";
  $("r1").textContent = "lr de 1e-3 para 1e-2 com queda a zero, ainda no Adam: " + fmt(bm) + " → " + fmt(D.mm.adam[0]) + ".";
  $("r2").textContent = "Adam → AdamW: " + fmt(D.mm.adam[0]) + " → " + fmt(D.mm.adamw[0]) + ". Controla a norma dos pesos, mas o F1 quase não muda.";
  $("r3").textContent = "Muon nas 2 matrizes ocultas: " + fmt(D.mm.adamw[0]) + " → " + fmt(mm) + ". Só ¼ do ganho, e exige a agenda que decai.";
  $("lD1").innerHTML = [
    "<b>Baseline:</b> ρ cai de 1e‑3 a <b>" + exp(D.rho_fim_baseline) + "</b> e a perda de treino termina em <b>" + fmt(D.perda_treino.baseline, 3) + "</b>: passo pequeno demais, a rede fica <b>subajustada</b>.",
    "<b>Adam lr 1e-2 + agenda:</b> ρ sobe a <b>" + exp(D.rho_pico.adam) + "</b> (6× a referência) e a queda do lr o leva a ~0; a perda termina em <b>" + fmt(D.perda_treino.adam, 3) + "</b>.",
    "<b>Duas forças:</b> lr grande no começo para andar mais; lr perto de 0 no fim para o modelo “assentar” no vale em vez de oscilar com o ruído do mini-lote."].map(s => "<li>" + s + "</li>").join("");
  $("tLim").innerHTML = "40 épocas: ganho de <b>+" + fmt(mm - bm) + "</b> (" + vezes(mm - bm, Math.max(ms, bs)) + "× o desvio). 200 épocas (" + nSem + "): <b>+" + fmt(m2 - b2) + "</b>" +
    (n2 > 1 ? " (" + vezes(m2 - b2, Math.max(s2, bs2)) + "× o desvio)" : "") + ". A rede ainda não saturou em 40 épocas, então parte do ganho do Muon é <b>velocidade</b>. " +
    (n2 > 1 ? "Mesmo assim o ganho continua maior que a variação entre sementes." : "Com 1 semente não dá para afirmar que supera o desvio.");
  const ins = D.instavel;
  $("tInst").innerHTML = "A perda <b>sobe a " + fmt(ins.pico, 1) + "</b> no passo " + ins.passo_pico.toLocaleString("pt-BR") + " e ‖θ‖ chega a " + Math.round(ins.theta_pico) + " (" + Math.round(ins.theta_final) +
    " no fim): <b>passo grande demais</b>, e o remédio seria um lr menor. A receita vencedora desce sem picos.";
}

/* ---------- notas e dados (gaveta) ---------- */
const NOTAS = [
`<h3>Página 1 · de onde partimos</h3>
<p><b>Dataset.</b> Trocamos porque o antigo era pequeno e saturado (98,6%): com pouca margem, nada do que a disciplina ensina aparecia. O Covertype (581 mil células, 7 classes desbalanceadas) tem espaço para evoluir a cada desafio.</p>
<p><b>Tema.</b> Cada linha é um terreno de 30×30 m (elevação, distâncias até a água, focos de incêndio, sombreamento, tipo de solo). A rede diz qual das 7 coberturas domina ali.</p>
<p><b>Rede.</b> 57 valores (as contínuas padronizadas) → 128 → 64 → 7. Camadas ocultas: multiplicam pelos pesos, somam o viés e aplicam a tanh. A saída só multiplica e soma e entrega 7 números (logits); o maior indica a cobertura. ~16 mil parâmetros, ~25 exemplos de treino por parâmetro.</p>
<p><b>Como a rede aprende (quadro de baixo).</b> A função de perda mede o erro: alta quando a rede dá pouca probabilidade à cobertura certa e baixa quando dá muita. O gradiente diz, para cada peso, para onde a perda cresce. O otimizador usa o gradiente para atualizar os pesos a cada passo; o que muda de um para outro é o tamanho e a direção do passo. É isso que o Desafio 3 compara.</p>
<p><b>Troca do ReLU pelo tanh (Desafio 2).</b> Testamos as ativações com 3 sementes cada; a ideia de testar o tanh veio das apresentações dos colegas. O tanh foi o melhor (3 desvios ou mais acima do GELU e do ReLU). É coerente com o Desafio 2: ele limita o sinal entre −1 e 1, o que ajuda a manter o treino estável. Se perguntarem a causa: medimos que ganha; a explicação é essa, mas não isolamos a causa.</p>
<p><b>Se perguntarem qual é a função de perda:</b> é a entropia cruzada, a perda padrão de classificação. Para cada exemplo ela olha só a probabilidade que a rede deu à cobertura certa e calcula <b>menos o log</b> dela: se a rede deu 90% à certa, a perda é pequena (~0,1); se deu 1%, é grande (~4,6). Por isso é alta quando a rede erra com confiança e baixa quando acerta. O Softmax, que transforma os 7 números em probabilidades, já vem embutido nela.</p>
<p><b>Baseline.</b> O enunciado pede a "receita atual do grupo"; como tínhamos acabado de trocar de dataset, não havia. Definimos o treino padrão: Adam, lr 1e-3, constante. Durante o desenvolvimento comparamos tudo com ela e buscamos uma receita melhor, que é o que o Desafio 3 pede.</p>
<h3>Glossário (os termos que saíram do slide)</h3>
<p><b>Otimizador:</b> a regra que usa o gradiente para atualizar os pesos (SGD, Adam, AdamW, Muon). <b>lr</b> (taxa de aprendizado): o tamanho do passo. <b>Agenda:</b> como o lr muda ao longo do treino. <b>Receita:</b> otimizador + lr + agenda; <b>baseline:</b> a receita de referência.</p>
<p><b>Passo / época:</b> 1 passo = 1 lote de 512 exemplos; 1 época = 795 passos (uma volta nos dados). <b>Validação / teste:</b> a validação decide as escolhas; o teste é medido uma única vez, no fim. <b>Semente:</b> sorteia a inicialização e a ordem dos lotes; o desvio entre sementes mostra quanto o resultado muda só por sorte.</p>
<p><b>F1:</b> nota de 0 a 1 do acerto numa classe (equilibra falsos alarmes e casos perdidos). <b>F1 macro:</b> média das 7 classes; as raras pesam igual às grandes. <b>ρ</b> = ‖Δθ‖/‖θ‖: tamanho relativo do passo (referência do enunciado: 1e‑3). <b>‖θ‖:</b> norma dos pesos.</p>
<p><b>Muon:</b> o passo das matrizes ocultas é ortogonalizado, então todas as direções andam o mesmo tanto. <b>Warmup + cosseno:</b> o lr sobe em 5% do treino e depois desce suave até 0.</p>`,
`<h3>Página 2 · o que o desafio pedia</h3>
<p><b>Comparar</b> (quadro de cima, à esquerda, para ler): SGD (com e sem Nesterov), Adam, AdamW e Muon; agendas constante, cosseno, warmup + cosseno e uma à escolha. O Muon atua só nas 2 matrizes ocultas; vieses e saída ficam no AdamW.</p>
<p><b>WSD (gráfico).</b> Escolhemos porque está nas leituras do enunciado e contrasta com o cosseno: o cosseno baixa o lr desde o começo; o WSD mantém o lr alto por 80% e só cai no final. Não ganhou, mas mostrou que o que importa é a queda do lr até perto de zero no fim.</p>
<p><b>40 épocas em tudo e 200 para confirmar (quadro da esquerda, embaixo).</b> Treinamos na CPU, sem GPU, e o Muon é o mais lento. O protocolo inteiro leva ~2 h com 40 épocas e passaria de 10 h com 200. Por isso 40 em todas as etapas, igual para todos, e depois 200 só na vencedora e na baseline, porque ficava a dúvida se a vantagem do Muon era só aprender mais rápido.</p>
<p><b>Outras decisões nossas.</b> Registro a cada 25 passos (o enunciado permite de 10 a 50: ~1.270 pontos por execução). Só o lr foi ajustado: momento e weight decay ficaram nos padrões, para saber o que causou o ganho. Sementes extras de AdamW e Adam: o top-3 da etapa 3 era todo Muon; sem elas, “Muon > AdamW” ficaria apoiado em 1 semente.</p>
<p><b>Protocolo e regras.</b> As 4 etapas e as regras do enunciado estão nos quadros de baixo, com quantas execuções foram feitas. Se perguntarem: o enunciado só autoriza reduzir o orçamento nas etapas 2 e 3; usamos 40 em todas e declaramos isso.</p>`,
`<h3>Página 3 · o que encontramos (~1 min)</h3>
<p><b>Receita e ganho (esquerda).</b> Muon lr 0,03 + AdamW lr 0,01 + warmup/cosseno: F1 0,9121 ± 0,0009 contra 0,8646 ± 0,0028; ~16× o desvio. Teste (usado uma vez): 0,9104 contra 0,8689.</p>
<p><b>Caminho até ela (meio e direita).</b> Etapa 2: o melhor lr de cada otimizador, todos no interior da grade. Etapa 3: com lr constante o Muon perde do AdamW; a vantagem exige a queda do lr. Etapa 4: confirmação com 3 sementes; cosseno puro empata com warmup + cosseno.</p>`,
`<h3>Página 4 · por que funciona e os limites (~1 min)</h3>
<p><b>Diagnóstico (meio).</b> Baseline: ρ cai a 5e-4 e a perda de treino termina em 0,201: passo pequeno demais, subajustada. Com lr 1e-2 + queda a zero: ρ sobe a 6e-3 e a perda termina em 0,132. Dois terços do ganho vêm daí.</p>
<p><b>Decomposição (esquerda).</b> lr + agenda +0,031; weight decay +0,004; Muon +0,013 (só ¼).</p>
<p><b>Limites (direita).</b> Com 200 épocas o ganho encolhe: parte do ganho é velocidade. A execução instável (SGD lr 3,0) mostra "passo grande demais": a perda sobe a 4,5 e ‖θ‖ a 817.</p>
<h3>Se perguntarem</h3>
<p><b>O Muon é a causa do ganho?</b> Só de ¼; ⅔ são lr + agenda.<br>
<b>Ajuste excessivo à validação?</b> O teste deu 0,9104 contra 0,9114 na validação.<br>
<b>Por que o warmup não ajudou?</b> Cosseno puro empata em 3 sementes.</p>`];

function tabela(cabecalho, linhas) {
  return "<table><tr>" + cabecalho.map(c => "<th>" + c + "</th>").join("") + "</tr>" +
    linhas.map(l => "<tr>" + l.map(c => "<td>" + c + "</td>").join("") + "</tr>").join("") + "</table>";
}
function montarDados() {
  const mm = D.mm, s = D.seeds;
  let h = "<h3>Ganho por etapa (F1 de validação, 3 sementes, 40 épocas)</h3>" + tabela(["Receita", "média ± desvio", "sementes 0 · 1 · 2"], [
    ["Baseline (Adam 1e-3, constante)", fmt(mm.baseline[0]) + " ± " + fmt(mm.baseline[1]), s.baseline.map(v => fmt(v)).join(" · ")],
    ["Adam lr 1e-2, warmup + cosseno", fmt(mm.adam[0]) + " ± " + fmt(mm.adam[1]), s.adam.map(v => fmt(v)).join(" · ")],
    ["AdamW lr 1e-2, warmup + cosseno", fmt(mm.adamw[0]) + " ± " + fmt(mm.adamw[1]), s.adamw.map(v => fmt(v)).join(" · ")],
    ["Muon lr 0,03, warmup + cosseno", fmt(mm.muon[0]) + " ± " + fmt(mm.muon[1]), s.muon.map(v => fmt(v)).join(" · ")]]);
  h += "<h3>200 épocas (" + D.e200.muon.length + " sementes)</h3>" + tabela(["", "média ± desvio", "sementes 0 · 1 · 2"], [
    ["Muon", fmt(D.e200.mm_muon[0]) + " ± " + fmt(D.e200.mm_muon[1]), D.e200.muon.map(v => fmt(v)).join(" · ")],
    ["Baseline", fmt(D.e200.mm_baseline[0]) + " ± " + fmt(D.e200.mm_baseline[1]), D.e200.baseline.map(v => fmt(v)).join(" · ")]]);
  h += "<h3>Teste (usado uma única vez)</h3>" + tabela(["", "F1 macro", "acurácia"], [
    ["Muon", fmt(D.teste.muon.f1), fmt(D.teste.muon.acc)], ["Baseline", fmt(D.teste.baseline.f1), fmt(D.teste.baseline.acc)]]);
  return h;
}

/* ---------- navegacao, cronometro, gaveta, tema ---------- */
const PAGINAS = [...document.querySelectorAll(".pagina")];
let pg = 0;
function irPara(i) {
  pg = Math.max(0, Math.min(PAGINAS.length - 1, i));
  PAGINAS.forEach((p, k) => p.classList.toggle("atual", k === pg));
  $("pos").textContent = (pg + 1) + " / " + PAGINAS.length;
  $("painelNotas").innerHTML = NOTAS[pg] || "";
  history.replaceState(null, "", "#p" + (pg + 1));
}
let rel = { ligado: false, rodando: false, t: 0 };
function atualizarRelogio() {
  const el = $("relogio"), resta = Math.max(0, 180 - rel.t);
  el.textContent = Math.floor(resta / 60) + ":" + String(Math.floor(resta % 60)).padStart(2, "0");
  el.classList.toggle("fim", resta === 0);
}
setInterval(() => { if (rel.ligado && rel.rodando) { rel.t += 0.25; atualizarRelogio(); } }, 250);
function alternarRelogio() {
  rel.ligado = !rel.ligado; rel.rodando = rel.ligado; rel.t = 0;
  $("relogio").style.display = rel.ligado ? "inline-block" : "none";
  if (rel.ligado) atualizarRelogio();
}
function alternarGaveta(aba) {
  const g = $("gaveta");
  const abrir = g.style.display !== "block" || (aba && g.dataset.aba !== aba);
  g.style.display = abrir ? "block" : "none";
  if (aba) { g.dataset.aba = aba;
    $("painelNotas").style.display = aba === "n" ? "block" : "none";
    $("painelDados").style.display = aba === "d" ? "block" : "none";
    $("aNotas").classList.toggle("sel", aba === "n"); $("aDados").classList.toggle("sel", aba === "d"); }
  ajustar();
}
function ajustar() {
  const g = $("gaveta").style.display === "block" ? 460 : 0;
  escala = Math.min((window.innerWidth - g) / 1600, window.innerHeight / 900);
  const p = $("palco");
  p.style.transform = "scale(" + escala + ")";
  p.style.left = Math.max(0, (window.innerWidth - g - 1600 * escala) / 2) + "px";
  p.style.top = Math.max(0, (window.innerHeight - 900 * escala) / 2) + "px";
}
function desenharTudo() {
  desenharRede(); desenharAtivacao(); desenharLegendas(); desenharCurvasAgenda(); desenharSementes(); desenharVarredura(); desenharAgendas(); desenharConfirmacao();
  desenharDecomposicao(); desenharD1(); desenharLimitacoes();
}
function trocarTema() {
  const r = document.documentElement;
  r.dataset.theme = r.dataset.theme === "dark" ? "light" : "dark";
  desenharTudo();
}
$("painelDados").innerHTML = montarDados();
$("bRel").onclick = alternarRelogio;
$("bNotas").onclick = () => alternarGaveta("n");
$("bTema").onclick = trocarTema;
$("bPdf").onclick = () => window.print();
$("bAnt").onclick = () => irPara(pg - 1);
$("bProx").onclick = () => irPara(pg + 1);
$("aNotas").onclick = () => alternarGaveta("n");
$("aDados").onclick = () => alternarGaveta("d");
window.addEventListener("keydown", ev => {
  const k = ev.key.toLowerCase();
  if (k === "arrowright" || k === "pagedown") irPara(pg + 1);
  else if (k === "arrowleft" || k === "pageup") irPara(pg - 1);
  else if (k === "home") irPara(0);
  else if (k === "end") irPara(PAGINAS.length - 1);
  else if (/^[1-4]$/.test(k)) irPara(+k - 1);
  else if (k === "t") alternarRelogio();
  else if (k === " ") { if (rel.ligado) { rel.rodando = !rel.rodando; ev.preventDefault(); } }
  else if (k === "r") { rel.t = 0; atualizarRelogio(); }
  else if (k === "n") alternarGaveta("n");
  else if (k === "v") alternarGaveta("d");
  else if (k === "d") trocarTema();
  else if (k === "p") window.print();
  else if (k === "?") { const a = $("ajuda"); a.style.display = a.style.display === "block" ? "none" : "block"; }
  else if (k === "f") { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen(); }
});
let temaAntes = null;
window.addEventListener("beforeprint", () => { temaAntes = document.documentElement.dataset.theme; document.documentElement.dataset.theme = "light"; desenharTudo(); });
window.addEventListener("afterprint", () => { document.documentElement.dataset.theme = temaAntes; desenharTudo(); });
window.addEventListener("resize", ajustar);
preencherTextos(); desenharTudo(); ajustar();
const inicio = /^#p([1-4])$/.exec(location.hash);
irPara(inicio ? +inicio[1] - 1 : 0);
if (location.hash === "#imprimir") { document.documentElement.dataset.theme = "light"; }
</script>
</body>
</html>
"""


def gerar_pdf(caminho_html, caminho_pdf):
    candidatos = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    navegador = next((c for c in candidatos if os.path.exists(c)), None)
    if navegador is None:
        print("Nenhum Edge/Chrome encontrado: abra slide.html e use Imprimir > Salvar como PDF.")
        return
    url = "file:///" + caminho_html.replace("\\", "/") + "#imprimir"
    subprocess.run([navegador, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={caminho_pdf}", "--virtual-time-budget=6000", url],
                   check=True, capture_output=True)
    print("slide.pdf salvo em", caminho_pdf)


def main():
    dados = montar_dados()
    html = MODELO.replace("/*__DADOS__*/", json.dumps(dados, ensure_ascii=False))
    caminho = os.path.join(PASTA, "slide.html")
    with open(caminho, "w", encoding="utf-8") as arq:
        arq.write(html)
    print("slide salvo em", caminho, f"({len(html) // 1024} KB)")
    if "--pdf" in sys.argv:
        gerar_pdf(caminho, os.path.join(PASTA, "slide.pdf"))


if __name__ == "__main__":
    main()
