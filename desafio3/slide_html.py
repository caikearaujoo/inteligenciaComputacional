"""
slide_html.py - gera a apresentacao do Desafio 3 (slide.html e slide.pdf)

Le os logs e o experimentos.csv e escreve um HTML AUTOSSUFICIENTE (sem internet,
sem bibliotecas): os graficos sao desenhados em SVG pelo proprio HTML, a partir
dos dados embutidos. Assim os numeros do slide nunca divergem dos dados.

Estrutura (7 paginas de 16:9):
  1. O slide de 3 minutos: receita vencedora, ganho +- desvio e um diagnostico
     lido nos sinais (os tres itens que o enunciado pede).
  2 a 7. Detalhe de cada ponto, para apoiar a apresentacao e as perguntas:
     protocolo, varredura de lr, agendas e confirmacao, decomposicao do ganho,
     o diagnostico e as limitacoes.

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

  <!-- ===================== 1 · O SLIDE DE 3 MINUTOS ===================== -->
  <section class="pagina atual" id="p1">
    <div class="cab">
      <h1 id="t1"></h1>
      <div class="sub">Forest Covertype · MLP 57→128→64→7 (16 mil parâmetros) · Desafio 3: o melhor treino para o nosso problema</div>
    </div>

    <div class="bloco" style="left:40px; top:104px; width:470px; height:756px;">
      <h2>Receita vencedora</h2>
      <div class="chips">
        <div class="chip"><span class="ponto" style="background:var(--muon)"></span><span><b>Muon</b>, lr 0,03, nas 2 matrizes ocultas (momento 0,95, Nesterov)</span></div>
        <div class="chip"><span class="ponto q" style="background:var(--adamw)"></span><span><b>AdamW</b>, lr 0,01, nos vieses e na camada de saída</span></div>
        <div class="chip"><span class="ponto" style="background:var(--ink2); border-radius:2px; height:5px; width:18px"></span><span>agenda <b>warmup (5%) + cosseno até 0</b></span></div>
        <div class="chip"><span class="ponto" style="background:transparent; border:2px solid var(--ink2)"></span><span>40 épocas = 31.800 passos, lote 512, igual para todos</span></div>
      </div>
      <div class="linha"></div>
      <div class="texto" id="tProt" style="font-size:17px"></div>
      <div class="linha"></div>
      <h2 style="font-size:18px">Para explicar</h2>
      <ul class="lista curta" style="font-size:15.5px; margin-top:4px">
        <li><b>F1 macro</b>: média do F1 das 7 classes; as classes raras pesam igual às grandes.</li>
        <li><b>Semente</b>: sorteia a inicialização e a ordem dos lotes. O desvio mostra quanto o resultado muda só por isso.</li>
        <li><b>ρ</b> = ‖Δθ‖/‖θ‖: tamanho relativo do passo (referência do enunciado: 1e‑3).</li>
        <li><b>Muon</b>: o passo das matrizes é ortogonalizado, então todas as direções andam o mesmo tanto.</li>
        <li><b>Warmup + cosseno</b>: o lr sobe em 5% do treino e desce suave até 0.</li>
      </ul>
    </div>

    <div class="bloco" style="left:530px; top:104px; width:520px; height:756px;">
      <h2>Ganho (F1 macro, validação)</h2>
      <div class="etapa">40 épocas · 3 sementes (protocolo)</div>
      <div class="grande" id="g40" style="margin-top:6px"></div>
      <div class="rotulo" style="margin-top:6px; font-size:16px" id="g40b"></div>
      <svg id="svgSem40" width="484" height="92" style="margin-top:6px"></svg>
      <div class="texto" id="g40r" style="margin-top:2px"></div>
      <div class="linha" style="margin:18px 0 14px"></div>
      <div class="etapa" id="e200rot"></div>
      <div class="medio" id="g200" style="margin-top:6px"></div>
      <div class="rotulo" style="margin-top:6px; font-size:16px" id="g200b"></div>
      <svg id="svgSem200" width="484" height="92" style="margin-top:6px"></svg>
      <div class="texto" id="g200r" style="margin-top:2px"></div>
      <div class="linha" style="margin:18px 0 14px"></div>
      <div class="texto" id="gTeste" style="margin-top:0"></div>
    </div>

    <div class="bloco" style="left:1070px; top:104px; width:490px; height:756px;">
      <h2>Diagnóstico lido nos sinais</h2>
      <div style="font-size:19px; font-weight:700; line-height:1.3">lr maior + queda a zero explica <span style="color:var(--adam)">⅔ do ganho</span> (+0,031)</div>
      <svg id="svgP1rho" width="450" height="270" style="margin-top:12px"></svg>
      <div class="texto" id="tD1a"></div>
      <div class="lido">Lido em: ρ (razão de atualização) e perda de treino. Detalhe na página 6.</div>
    </div>
    <div class="npag">1 / 7</div>
  </section>

  <!-- ===================== 2 · PROTOCOLO ===================== -->
  <section class="pagina" id="p2">
    <div class="cab"><h1>Como comparamos: o protocolo do enunciado, em 4 etapas</h1>
      <div class="sub" id="sub2"></div></div>
    <div class="bloco" style="left:40px; top:104px; width:362px; height:338px;" id="et1"></div>
    <div class="bloco" style="left:426px; top:104px; width:362px; height:338px;" id="et2"></div>
    <div class="bloco" style="left:812px; top:104px; width:362px; height:338px;" id="et3"></div>
    <div class="bloco" style="left:1198px; top:104px; width:362px; height:338px;" id="et4"></div>
    <div class="bloco" style="left:40px; top:458px; width:748px; height:402px;">
      <h2>Regras do enunciado e como cumprimos</h2>
      <ul class="lista curta" id="regras"></ul>
    </div>
    <div class="bloco" style="left:812px; top:458px; width:748px; height:402px;">
      <h2>Escolhas nossas, declaradas</h2>
      <ul class="lista curta" id="escolhas"></ul>
    </div>
    <div class="npag">2 / 7</div>
  </section>

  <!-- ===================== 3 · VARREDURA DE LR ===================== -->
  <section class="pagina" id="p3">
    <div class="cab"><h1>Etapa 2: o melhor lr de cada otimizador, todos no interior da grade</h1>
      <div class="sub">F1 macro na validação × lr de partida · warmup + cosseno · 40 épocas · 1 semente por ponto · no Muon, o lr das matrizes (a parte AdamW fica em 0,01)</div></div>
    <div class="bloco" style="left:40px; top:104px; width:1000px; height:756px;">
      <svg id="svgVar" width="964" height="722"></svg>
    </div>
    <div class="bloco" style="left:1060px; top:104px; width:500px; height:756px;">
      <h2>Melhor lr de cada otimizador</h2>
      <table class="t" id="tabVar"></table>
      <ul class="lista curta" id="lVar" style="margin-top:14px"></ul>
    </div>
    <div class="npag">3 / 7</div>
  </section>

  <!-- ===================== 4 · AGENDAS E CONFIRMACAO ===================== -->
  <section class="pagina" id="p4">
    <div class="cab"><h1>Etapas 3 e 4: as 4 agendas e a confirmação com 3 sementes</h1>
      <div class="sub">Etapa 3 com 1 semente (AdamW lr 0,01 e Muon lr 0,03) · etapa 4 com 3 sementes nas melhores combinações, 40 épocas</div></div>
    <div class="bloco" style="left:40px; top:104px; width:740px; height:756px;">
      <h2>As 4 agendas (fator sobre o lr de partida)</h2>
      <svg id="svgAgFator" width="704" height="236"></svg>
      <div class="linha"></div>
      <h2>F1 por agenda · 1 semente</h2>
      <svg id="svgAgDots" width="704" height="206"></svg>
      <div class="texto" id="tAg"></div>
    </div>
    <div class="bloco" style="left:800px; top:104px; width:760px; height:756px;">
      <h2>Confirmação · F1 por semente (média ± desvio)</h2>
      <svg id="svgConf" width="724" height="470"></svg>
      <ul class="lista curta" id="lConf"></ul>
    </div>
    <div class="npag">4 / 7</div>
  </section>

  <!-- ===================== 5 · DECOMPOSICAO ===================== -->
  <section class="pagina" id="p5">
    <div class="cab"><h1 id="t5"></h1>
      <div class="sub">F1 macro de validação, média de 3 sementes ± desvio · cada degrau muda uma coisa só</div></div>
    <div class="bloco" style="left:40px; top:104px; width:900px; height:756px;">
      <svg id="svgDec" width="864" height="722"></svg>
    </div>
    <div class="bloco" style="left:960px; top:104px; width:600px; height:236px;">
      <h2>1 · lr maior + queda a zero<span class="tag" style="color:var(--adam)" id="tg1"></span></h2>
      <div class="texto" id="r1"></div>
      <div class="lido">Diagnóstico na página 6, lido em ρ e perda de treino</div>
    </div>
    <div class="bloco" style="left:960px; top:356px; width:600px; height:236px;">
      <h2>2 · Weight decay<span class="tag" style="color:var(--adamw)" id="tg2"></span></h2>
      <div class="texto" id="r2"></div>
      
    </div>
    <div class="bloco" style="left:960px; top:608px; width:600px; height:252px;">
      <h2>3 · Muon<span class="tag" style="color:var(--muon)" id="tg3"></span></h2>
      <div class="texto" id="r3"></div>
      
    </div>
    <div class="npag">5 / 7</div>
  </section>

  <!-- ===================== 6 · DIAGNOSTICO 1 ===================== -->
  <section class="pagina" id="p6">
    <div class="cab"><h1>Diagnóstico 1 · lr maior + queda a zero (+0,031, ⅔ do ganho)</h1>
      <div class="sub">Baseline (Adam 1e-3, constante) contra Adam lr 1e-2 com warmup + cosseno · mesmo eixo de passos: ler cada coluna vertical nos três gráficos</div></div>
    <div class="bloco" style="left:40px; top:104px; width:1000px; height:756px; padding-top:10px;">
      <div class="rotulo">ρ = ‖θ₊ − θ‖/‖θ‖ (razão de atualização, escala log)</div>
      <svg id="svgD1a" width="964" height="226"></svg>
      <div class="rotulo">perda de treino (média móvel por registro, escala log)</div>
      <svg id="svgD1b" width="964" height="226"></svg>
      <div class="rotulo">lr efetivo η do passo</div>
      <svg id="svgD1c" width="964" height="226"></svg>
    </div>
    <div class="bloco" style="left:1060px; top:104px; width:500px; height:756px;">
      <h2>O que os sinais mostram</h2>
      <ul class="lista curta" id="lD1"></ul>
      <div class="linha"></div>
      <h2 style="font-size:18px">Como ler (tabela do enunciado)</h2>
      <div class="pequeno">ρ ≈ 1e-3 é a referência. Passo pequeno demais deixa o treino subajustado: o remédio é η maior. A queda brusca no fim do cosseno é esperada: o decaimento final "assenta" no vale.</div>
    </div>
    <div class="npag">6 / 7</div>
  </section>

  <!-- ===================== 9 · LIMITACOES ===================== -->
  <section class="pagina" id="p7">
    <div class="cab"><h1>Limitações e execução instável</h1>
      <div class="sub">O que o resultado não prova, e o que a execução que falhou mostra nos sinais</div></div>
    <div class="bloco" style="left:40px; top:104px; width:600px; height:756px;">
      <h2>Mais treino: o ganho encolhe</h2>
      <svg id="svgLim" width="564" height="300"></svg>
      <div class="texto" id="tLim"></div>
    </div>
    <div class="bloco" style="left:660px; top:104px; width:900px; height:756px;">
      <h2>Execução instável: SGD, lr 3,0</h2>
      <svg id="svgInst" width="864" height="360"></svg>
      <div class="texto" id="tInst"></div>
    </div>
    <div class="npag">7 / 7</div>
  </section>
</div>

<div id="barra" class="ui">
  <button id="bAnt" title="←">◀</button>
  <span id="pos" style="font-size:13px">1 / 7</span>
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
<div id="ajuda" class="ui">← → páginas · 1 a 7 vai à página · T cronômetro (espaço pausa, R zera) · N notas · V dados · D tema · F tela cheia · P imprimir/PDF · ? esta ajuda</div>

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

/* ---------- pagina 1: sementes 40 e 200 epocas ---------- */
function desenharSementes() {
  const nb = D.seeds.baseline, nm = D.seeds.muon;
  const linha = (nome, c, forma, vals) => { const [mu, sd] = [vals.reduce((a, b) => a + b, 0) / vals.length,
    Math.sqrt(vals.reduce((a, b) => a + (b - vals.reduce((x, y) => x + y, 0) / vals.length) ** 2, 0) / vals.length)];
    return { nome, cor: c, forma, vals, mu, sd }; };
  sementesGen($("svgSem40"), { ml: 80, mr: 24, lo: 0.855, hi: 0.93, ticks: [0.86, 0.88, 0.90, 0.92], cap: "cada ponto = 1 semente (F1 de validação)",
    linhas: [linha("Muon", "--muon", "c", nm), linha("Baseline", "--neutro", "t", nb)] });
  sementesGen($("svgSem200"), { ml: 80, mr: 24, lo: 0.855, hi: 0.93, ticks: [0.86, 0.88, 0.90, 0.92], cap: "cada ponto = 1 semente (F1 de validação)",
    linhas: [linha("Muon", "--muon", "c", D.e200.muon), linha("Baseline", "--neutro", "t", D.e200.baseline)] });
}

/* ---------- pagina 3: varredura de lr ---------- */
function desenharVarredura() {
  const svg = $("svgVar"); svg.innerHTML = "";
  const W = 964, H = 722, m = { l: 74, r: 26, t: 24, b: 62 };
  const lx0 = Math.log10(2e-4), lx1 = Math.log10(6), y0 = 0.55, y1 = 0.95;
  const sx = v => m.l + (Math.log10(v) - lx0) / (lx1 - lx0) * (W - m.l - m.r);
  const sy = v => m.t + (H - m.t - m.b) * (1 - (v - y0) / (y1 - y0));
  for (let v = 0.55; v <= 0.951; v += 0.05) {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "grade" }, svg);
    texto(svg, m.l - 10, sy(v) + 4, fmt(v, 2), { "text-anchor": "end", class: "tick" });
  }
  [[1e-3, "1e-3"], [1e-2, "1e-2"], [1e-1, "1e-1"], [1, "1"]].forEach(([v, s]) => {
    svgEl("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "grade" }, svg);
    texto(svg, sx(v), H - m.b + 22, s, { "text-anchor": "middle", class: "tick" });
  });
  texto(svg, m.l + (W - m.l - m.r) / 2, H - 14, "lr de partida (escala log)", { "text-anchor": "middle", class: "tick" });
  svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "eixo" }, svg);
  const bm = D.mm.baseline[0];
  svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(bm), y2: sy(bm), stroke: cor("--ink2"), "stroke-width": 1.4, "stroke-dasharray": "6 5" }, svg);
  texto(svg, W - m.r, sy(bm) - 7, "baseline: " + fmt(bm), { "text-anchor": "end", class: "tick" });
  Object.entries(D.varredura).forEach(([k, pts]) => {
    const [nome, c, forma] = OPT[k];
    let d = "";
    pts.forEach((p, i) => { d += (i ? "L" : "M") + sx(p.lr).toFixed(1) + " " + sy(p.f1).toFixed(1); });
    svgEl("path", { d, fill: "none", stroke: cor(c), "stroke-width": 2.4, opacity: .9 }, svg);
    let melhor = pts[0];
    pts.forEach(p => {
      if (p.f1 > melhor.f1) melhor = p;
      const mk = marca(svg, forma, sx(p.lr), sy(p.f1), 6.5, cor(c));
      mk.addEventListener("mousemove", ev => mostrarDica("<b>" + nome + "</b><br>lr " + String(p.lr).replace(".", ",") + "<br>F1 " + fmt(p.f1), ev));
      mk.addEventListener("mouseleave", esconderDica);
    });
    svgEl("circle", { cx: sx(melhor.lr), cy: sy(melhor.f1), r: 15, fill: "none", stroke: cor("--ink"), "stroke-width": 2 }, svg);
  });
  /* legenda */
  Object.entries(OPT).forEach(([k, [nome, c, forma]], i) => {
    marca(svg, forma, m.l + 458, m.t + 380 + i * 26, 7, cor(c));
    texto(svg, m.l + 474, m.t + 385 + i * 26, nome, { class: "rot", fill: cor("--ink") });
  });
  texto(svg, m.l + 450, m.t + 380 + 5 * 26 + 12, "◯ = melhor lr de cada um", { class: "tick" });
  /* colapso do SGD lr 3,0 */
  const pi = D.varredura.sgd.find(p => p.lr === 3);
  if (pi) texto(svg, sx(3) - 14, sy(pi.f1) + 36, "SGD lr 3,0 colapsou", { "text-anchor": "end", class: "rot", fill: cor("--sgd") });
}

/* ---------- pagina 4: agendas ---------- */
function fatorAgenda(ag, t) {
  if (ag === "constante") return 1;
  if (ag === "cosseno") return 0.5 * (1 + Math.cos(Math.PI * t));
  if (ag === "warmup_cosseno") return t < 0.05 ? t / 0.05 : 0.5 * (1 + Math.cos(Math.PI * (t - 0.05) / 0.95));
  return t < 0.05 ? t / 0.05 : t < 0.8 ? 1 : 1 - (t - 0.8) / 0.2;
}
const AGENDAS = ["constante", "cosseno", "warmup_cosseno", "wsd"];
const AG = { constante: ["constante", "--neutro"], cosseno: ["cosseno", "--ag2"], warmup_cosseno: ["warmup + cosseno", "--ink"], wsd: ["WSD (warmup, estável, queda)", "--ag4"] };
function desenharAgendas() {
  const series = Object.entries(AG).map(([ag, [nome, c]]) => {
    const x = [], y = [];
    for (let i = 0; i <= 160; i++) { x.push(i / 160 * 31800); y.push(fatorAgenda(ag, i / 160)); }
    return { nome, cor: c, x, y };
  });
  graficoLinhas($("svgAgFator"), { x0: 0, x1: 31800, y0: 0, y1: 1.1, log: false, yticks: [0, 0.5, 1], xticks: [0, 10000, 20000, 30000],
    yfmt: v => String(v).replace(".", ","), tip: v => "fator " + fmt(v, 2), series,
    });
  const lg = $("svgAgFator");
  Object.values(AG).forEach(([nome, c], i) => {
    svgEl("line", { x1: 396, x2: 418, y1: 52 + i * 22, y2: 52 + i * 22, stroke: cor(c), "stroke-width": 3 }, lg);
    texto(lg, 426, 56 + i * 22, nome, { class: "tick", fill: cor("--ink") });
  });
  /* F1 por agenda */
  const svg = $("svgAgDots"); svg.innerHTML = "";
  const W = 704, H = 206, m = { l: 150, r: 130, t: 22, b: 26 }, lo = 0.80, hi = 0.92;
  const sx = v => m.l + (v - lo) / (hi - lo) * (W - m.l - m.r);
  const rowH = (H - m.t - m.b) / 4;
  [0.80, 0.84, 0.88, 0.92].forEach(t => {
    svgEl("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b, class: "grade" }, svg);
    texto(svg, sx(t), H - m.b + 17, fmt(t, 2), { "text-anchor": "middle", class: "tick" });
  });
  texto(svg, m.l, 12, "■ AdamW lr 0,01   ● Muon lr 0,03", { class: "tick" });
  AGENDAS.forEach((ag, i) => {
    const y = m.t + (i + 0.5) * rowH, a = D.agendas.adamw[ag], mu = D.agendas.muon[ag];
    texto(svg, 0, y + 5, AG[ag][0].replace(" (warmup, estável, queda)", ""), { class: "rot", fill: cor("--ink") });
    svgEl("line", { x1: sx(a), x2: sx(mu), y1: y, y2: y, stroke: cor("--ink2"), "stroke-width": 1.4 }, svg);
    marca(svg, "q", sx(a), y, 7, cor("--adamw")); marca(svg, "c", sx(mu), y, 7, cor("--muon"));
    texto(svg, W, y + 5, fmt(a, 3) + "  ·  " + fmt(mu, 3), { "text-anchor": "end", class: "rot", fill: cor("--ink") });
  });
}

/* ---------- decomposicao do ganho ---------- */
function desenharDecomposicao() {
  const svg = $("svgDec"); svg.innerHTML = "";
  const W = 864, H = 722, m = { l: 66, r: 20, t: 40, b: 70 }, lo = 0.855, hi = 0.93;
  const sx = i => m.l + (i + 0.5) / 4 * (W - m.l - m.r);
  const sy = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
  [0.86, 0.88, 0.90, 0.92].forEach(t => {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "grade" }, svg);
    texto(svg, m.l - 8, sy(t) + 4, fmt(t, 2), { "text-anchor": "end", class: "tick" });
  });
  const passos = [
    ["Baseline", "Adam 1e-3, constante", "baseline", "--neutro", "t"],
    ["+ lr 1e-2 + agenda", "Adam, warmup + cosseno", "adam", "--adam", "t"],
    ["+ weight decay", "AdamW", "adamw", "--adamw", "q"],
    ["+ Muon", "nas matrizes ocultas", "muon", "--muon", "c"],
  ];
  let d = "";
  passos.forEach((p, i) => { d += (i ? "L" : "M") + sx(i) + " " + sy(D.mm[p[2]][0]); });
  svgEl("path", { d, fill: "none", stroke: cor("--ink2"), "stroke-width": 1.6 }, svg);
  passos.forEach((p, i) => {
    const [mu, sd] = D.mm[p[2]], x = sx(i), y = sy(mu);
    svgEl("line", { x1: x, x2: x, y1: sy(mu - sd), y2: sy(mu + sd), stroke: cor("--ink"), "stroke-width": 2 }, svg);
    D.seeds[p[2]].forEach(v => svgEl("circle", { cx: x + 20, cy: sy(v), r: 3.5, fill: cor(p[3]), opacity: .55 }, svg));
    const mk = marca(svg, p[4], x, y, 11, cor(p[3]));
    texto(svg, x, y - 24, fmt(mu), { "text-anchor": "middle", class: "rot", fill: cor("--ink"), "font-size": 17 });
    if (i) texto(svg, x + 34, y + 30, "(+" + fmt(mu - D.mm[passos[i - 1][2]][0]) + ")", { "text-anchor": "start", class: "tick", fill: cor("--ink2"), "font-size": 15 });
    texto(svg, x, H - 36, p[0], { "text-anchor": "middle", class: "rot", fill: cor("--ink"), "font-size": 16 });
    texto(svg, x, H - 16, p[1], { "text-anchor": "middle", class: "tick" });
    mk.addEventListener("mousemove", ev => mostrarDica("<b>" + p[1] + "</b><br>F1 " + fmt(mu) + " ± " + fmt(sd) + "<br>sementes: " + D.seeds[p[2]].map(v => fmt(v)).join(" · "), ev));
    mk.addEventListener("mouseleave", esconderDica);
  });
  texto(svg, m.l, 20, "pontos pequenos = as 3 sementes · barra = ± desvio", { class: "tick" });
}

/* ---------- confirmacao (3 sementes) ---------- */
function desenharConfirmacao() {
  sementesGen($("svgConf"), { ml: 250, mr: 150, lo: 0.855, hi: 0.925, ticks: [0.86, 0.88, 0.90, 0.92], cap: "cada ponto = 1 semente (F1 de validação)", valores: true,
    linhas: D.confirmacao.map((c, i) => ({ ...c, destaque: i === 0 })) });
}

/* ---------- diagnostico 1: rho, perda e lr ---------- */
function desenharRho(id) {
  graficoLinhas($(id), {
    x0: 0, x1: 31800, y0: 1e-5, y1: 0.05, log: true, yticks: [1e-5, 1e-4, 1e-3, 1e-2], xticks: [0, 10000, 20000, 30000],
    yfmt: v => v.toExponential(0), tip: v => "ρ = " + v.toExponential(1),
    ref: [{ y: 1e-3, texto: "referência do enunciado 1e-3" }],
    series: [{ nome: "Baseline (Adam 1e-3, constante)", cor: "--neutro", x: D.rho.baseline.x, y: D.rho.baseline.y },
             { nome: "Adam lr 1e-2 + warmup/cosseno", cor: "--adam", x: D.rho.adam.x, y: D.rho.adam.y }],
    rotulos: [{ x: 9500, y: 1.9e-4, texto: "baseline: ρ ~1e-3 → 5e-4", cor: "--ink2" },
              { x: 9800, y: 1.35e-2, texto: "lr 1e-2: pico 6e-3, depois → 0", cor: "--adam" }]
  });
}
function desenharD1() {
  desenharRho("svgD1a");
  graficoLinhas($("svgD1b"), {
    x0: 0, x1: 31800, y0: 0.1, y1: 2.5, log: true, yticks: [0.1, 0.2, 0.5, 1, 2], xticks: [0, 10000, 20000, 30000], semX: true,
    yfmt: v => String(v).replace(".", ","), tip: v => "perda = " + fmt(v, 3),
    series: [{ nome: "Baseline", cor: "--neutro", x: D.perda.baseline.x, y: D.perda.baseline.y },
             { nome: "Adam lr 1e-2", cor: "--adam", x: D.perda.adam.x, y: D.perda.adam.y }],
    rotulos: [{ x: 31300, y: D.perda_treino.baseline * 1.35, texto: "baseline: " + fmt(D.perda_treino.baseline, 3), cor: "--ink2", ancora: "end" },
              { x: 31300, y: D.perda_treino.adam * 0.72, texto: "lr 1e-2: " + fmt(D.perda_treino.adam, 3), cor: "--adam", ancora: "end" }]
  });
  graficoLinhas($("svgD1c"), {
    x0: 0, x1: 31800, y0: 0, y1: 0.0115, log: false, yticks: [0, 0.005, 0.01], xticks: [0, 10000, 20000, 30000],
    yfmt: v => String(v).replace(".", ","), tip: v => "η = " + v.toExponential(2),
    series: [{ nome: "Baseline", cor: "--neutro", x: D.lr.baseline.x, y: D.lr.baseline.y },
             { nome: "Adam lr 1e-2", cor: "--adam", x: D.lr.adam.x, y: D.lr.adam.y }],
    rotulos: [{ x: 5200, y: 0.0016, texto: "constante 1e-3", cor: "--ink2" }, { x: 12500, y: 0.0098, texto: "warmup 5% → cosseno até 0", cor: "--adam" }]
  });
}
/* ---------- pagina 7: limitacoes ---------- */
function desenharLimitacoes() {
  const [m4, s4] = D.mm.muon, [b4, bs4] = D.mm.baseline, [m2, s2] = D.e200.mm_muon, [b2, bs2] = D.e200.mm_baseline;
  const n = D.e200.muon.length;
  const ln = (nome, c, forma, vals, sem) => { const mu = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mu) ** 2, 0) / vals.length); return { nome, cor: c, forma, vals, mu, sd, sem }; };
  sementesGen($("svgLim"), { ml: 150, mr: 140, lo: 0.855, hi: 0.935, ticks: [0.86, 0.88, 0.90, 0.92], cap: "cada ponto = 1 semente (F1 de validação)", valores: true,
    linhas: [ln("Muon · 40 épocas", "--muon", "c", D.seeds.muon), ln("Baseline · 40 épocas", "--neutro", "t", D.seeds.baseline),
             ln("Muon · 200 épocas", "--muon", "c", D.e200.muon), ln("Baseline · 200 épocas", "--neutro", "t", D.e200.baseline)] });
  graficoLinhas($("svgInst"), {
    x0: 0, x1: 31800, y0: 0.1, y1: 6, log: true, yticks: [0.1, 0.2, 0.5, 1, 2, 5], xticks: [0, 10000, 20000, 30000],
    yfmt: v => String(v).replace(".", ","), tip: v => "perda = " + fmt(v, 3),
    series: [{ nome: "SGD lr 3,0", cor: "--sgd", x: D.instavel.perda.x, y: D.instavel.perda.y },
             { nome: "Muon (a vencedora)", cor: "--muon", x: D.instavel.perda_muon.x, y: D.instavel.perda_muon.y }],
    rotulos: [{ x: D.instavel.passo_pico + 1500, y: D.instavel.pico * 0.9, texto: "pico " + fmt(D.instavel.pico, 1), cor: "--sgd" },
              { x: 14000, y: 0.2, texto: "Muon", cor: "--muon" }]
  });
}

/* ---------- textos que dependem dos dados ---------- */
function preencherTextos() {
  const [mm, ms] = D.mm.muon, [bm, bs] = D.mm.baseline, c = D.contagem;
  const [m2, s2] = D.e200.mm_muon, [b2, bs2] = D.e200.mm_baseline, n2 = D.e200.muon.length;
  const pct = v => (v * 100).toFixed(1).replace(".", ",") + "%";
  $("t1").textContent = "Muon + queda do lr: +" + fmt(mm - bm, 3) + " de F1 sobre a baseline, e o que os sinais mostram";
  $("tProt").innerHTML = "Comparados: SGD (± Nesterov), Adam, AdamW e Muon × 4 agendas, com o lr ajustado em cada um. <b>" + D.n_exec + " execuções, " +
    (D.n_diverg === 0 ? "nenhuma divergiu" : D.n_diverg + " divergiram") + ".</b><br><br><b>Adotada no modelo final do projeto</b> (modelo.py): teste " + pct(D.teste.muon.acc) + ".";
  $("g40").textContent = fmt(mm) + " ± " + fmt(ms);
  $("g40b").innerHTML = "contra <b class='num'>" + fmt(bm) + " ± " + fmt(bs) + "</b> da baseline (Adam 1e-3, constante)";
  $("g40r").innerHTML = "<b>+" + fmt(mm - bm) + "</b> = " + Math.floor((mm - bm) / Math.max(ms, bs)) + "× o desvio entre sementes";
  $("e200rot").textContent = "200 épocas · " + n2 + (n2 === 1 ? " semente" : " sementes") + " (complemento)";
  $("g200").textContent = fmt(m2) + (n2 > 1 ? " ± " + fmt(s2) : "");
  $("g200b").innerHTML = "contra <b class='num'>" + fmt(b2) + (n2 > 1 ? " ± " + fmt(bs2) : "") + "</b> da baseline";
  $("g200r").innerHTML = "<b>+" + fmt(m2 - b2) + "</b>" + (n2 > 1 ? " = " + Math.floor((m2 - b2) / Math.max(s2, bs2)) + "× o desvio entre sementes" : " (1 semente: sem desvio)");
  $("gTeste").innerHTML = "<b>Teste (usado uma única vez):</b> F1 <b class='num'>" + fmt(D.teste.muon.f1) + "</b> contra " + fmt(D.teste.baseline.f1) +
    " · acurácia " + pct(D.teste.muon.acc) + " contra " + pct(D.teste.baseline.acc);
  const exp = v => v.toExponential(0).replace("-", "‑");
  $("tD1a").innerHTML = "A baseline mexe pouco: ρ cai de 1e‑3 a <b>" + exp(D.rho_fim_baseline) + "</b> e a perda de treino fica em <b>" + fmt(D.perda_treino.baseline, 3) +
    "</b> (subajustada). Com lr 0,01 o ρ chega a <b>" + exp(D.rho_pico.adam) + "</b> e a agenda o leva a ~0: perda de treino <b>" + fmt(D.perda_treino.adam, 3) + "</b>.";
  /* pagina 2 */
  $("sub2").textContent = "Mesmo orçamento para todas as comparações: 40 épocas × 795 passos = 31.800 passos, lote 512 · validação decide, teste só no final · " + c.total + " execuções reportadas";
  const cartao = (id, et, titulo, corpo, res) => { $(id).innerHTML = "<div class='etapa'>" + et + "</div><h2 style='margin-top:4px'>" + titulo + "</h2><div class='texto' style='font-size:16px'>" + corpo +
    "</div><div class='linha'></div><div class='texto' style='font-size:16px; margin-top:0'>" + res + "</div>"; };
  const melhor = k => D.varredura[k].reduce((a, b) => b.f1 > a.f1 ? b : a);
  cartao("et1", "Etapa 1", "Baseline", "A receita atual do grupo: Adam lr 1e-3, agenda constante, 3 sementes.", "<b>" + c.etapa1 + " execuções</b><br>F1 " + fmt(bm) + " ± " + fmt(bs));
  cartao("et2", "Etapa 2", "Varredura de lr", "Cada otimizador (SGD, Nesterov, Adam, AdamW, Muon) com 5 a 7 valores de lr em escala log, warmup + cosseno, 1 semente. O SGD foi estendido até 3,0 porque o melhor caiu na borda.",
    "<b>" + c.etapa2 + " execuções</b><br>Melhor: Muon lr " + String(melhor("muon").lr).replace(".", ",") + " (" + fmt(melhor("muon").f1) + ")");
  cartao("et3", "Etapa 3", "Agendas", "As 4 agendas (constante, cosseno, warmup + cosseno, WSD) para o AdamW (lr 0,01) e o Muon (lr 0,03), cada um com o seu melhor lr, 1 semente.",
    "<b>" + c.etapa3 + " execuções novas</b><br>Vence warmup + cosseno nos dois");
  cartao("et4", "Etapa 4", "Confirmação", "As melhores combinações com 3 sementes, incluindo AdamW e Adam com o mesmo lr e agenda (para separar o efeito de cada degrau). Complemento declarado: vencedora e baseline também com 200 épocas.",
    "<b>" + (c.etapa4_40 + c.etapa4_200) + " execuções</b> (" + c.etapa4_40 + " de 40 épocas, " + c.etapa4_200 + " de 200)<br>Vencedora: " + fmt(mm) + " ± " + fmt(ms));
  $("regras").innerHTML = [
    "<span class='ok'>✓</span> <b>lr ajustado por otimizador</b>: varredura própria de cada um (e do Muon sobre o melhor lr do AdamW).",
    "<span class='ok'>✓</span> <b>Mesmo orçamento</b> (passos e lote) em todas as comparações.",
    "<span class='ok'>✓</span> <b>Seleção na validação</b>; o teste foi usado <b>uma única vez</b>, no fim.",
    "<span class='ok'>✓</span> <b>Tudo reportado</b>, inclusive o que falhou: " + (D.n_diverg === 0 ? "nenhuma execução divergiu; o SGD com lr 3,0 colapsou (F1 " + fmt(D.instavel.f1, 2) + ")" : D.n_diverg + " divergiram") + ".",
    "<span class='ok'>✓</span> <b>Só há vencedor se a diferença supera a variação entre sementes</b>: +" + fmt(mm - bm, 3) + " contra desvios de " + fmt(ms, 4) + " e " + fmt(bs, 4) + ".",
    "<span class='ok'>✓</span> <b>Muon incluído</b> (<code>torch.optim.Muon</code>, só nas matrizes 2-D ocultas; vieses e saída no AdamW).",
    "<span class='ok'>✓</span> <b>Reprodutível</b>: cada linha do <code>experimentos.csv</code> traz o comando exato."].map(s => "<li style='list-style:none; margin-left:-22px'>" + s + "</li>").join("");
  $("escolhas").innerHTML = [
    "<b>40 épocas em todas as etapas</b>, inclusive baseline e confirmação, por limite de CPU: o Muon custa ~" + D.e200.custo.toFixed(1).replace(".", ",") + "× mais por época. O enunciado só autoriza reduzir nas etapas 2–3, então declaramos e fizemos o complemento de 200 épocas com " + n2 + (n2 === 1 ? " semente" : " sementes") + ".",
    "<b>WSD</b> como 4ª agenda (a \"à escolha\"): warmup 5%, estável até 80% e queda linear até 0.",
    "<b>Registro a cada 25 passos</b> (o enunciado permite de 10 a 50), global e por camada.",
    "<b>Só o lr foi ajustado</b>: momento (SGD 0,9; Muon 0,95 com Nesterov) e weight decay (AdamW e Muon 0,01; Adam sem) ficaram fixos.",
    "<b>Complemento declarado</b>: sementes 1 e 2 do AdamW e do Adam, porque o top-3 foi todo Muon e \"Muon > AdamW\" ficaria sustentado por 1 semente."].map(s => "<li>" + s + "</li>").join("");

  /* pagina 3 */
  const ordem = ["muon", "adamw", "adam", "sgd_nesterov", "sgd"];
  $("tabVar").innerHTML = "<tr><th>otimizador</th><th>melhor lr</th><th>F1</th></tr>" + ordem.map(k => {
    const b = melhor(k); return "<tr><td><b style='color:var(" + OPT[k][1] + ")'>●</b> " + OPT[k][0] + "</td><td>" + String(b.lr).replace(".", ",") + "</td><td>" + fmt(b.f1) + "</td></tr>"; }).join("");
  $("lVar").innerHTML = [
    "O Muon (<b>" + fmt(melhor("muon").f1) + "</b>) está acima de todos, e todos os melhores lr ficam <b>no interior</b> da grade.",
    "Adam e AdamW pedem lr ~10× menor que o SGD, e o Muon (matrizes) fica no meio.",
    "O SGD com lr 3,0 <b>colapsou</b> (F1 " + fmt(D.instavel.f1, 2) + "): é a execução instável (página 7).",
    "1 semente por ponto: diferenças menores que ~0,003 não separam efeito de ruído. A confirmação com 3 sementes está na página 4."].map(s => "<li>" + s + "</li>").join("");

  /* pagina 4 */
  $("tAg").innerHTML = "Com lr <b>constante</b> o Muon perde do AdamW (<b>" + fmt(D.agendas.muon.constante, 3) + "</b> contra <b>" + fmt(D.agendas.adamw.constante, 3) +
    "</b>): a vantagem do Muon exige a queda do lr. O WSD ficou abaixo do warmup + cosseno nos dois.";
  const cf = D.confirmacao, mu1 = cf[0], mu2 = cf[1], mu3 = cf[2];
  $("lConf").innerHTML = [
    "<b>Vencedora:</b> Muon lr 0,03 com warmup + cosseno, <b>" + fmt(mu1.mu) + " ± " + fmt(mu1.sd) + "</b>, contra " + fmt(cf[6].mu) + " ± " + fmt(cf[6].sd) + " da baseline.",
    "Cosseno puro empata com warmup + cosseno (" + fmt(mu2.mu) + " ± " + fmt(mu2.sd) + "): a diferença de " + fmt(Math.abs(mu1.mu - mu2.mu), 4) + " é menor que o desvio, então o <b>warmup não fez diferença</b> aqui.",
    "O WSD fica " + fmt(mu1.mu - mu3.mu, 4) + " abaixo e é mais instável entre sementes (desvio " + fmt(mu3.sd) + ").",
    "AdamW (" + fmt(cf[3].mu) + ") e Adam (" + fmt(cf[5].mu) + ") com o mesmo lr e agenda: o weight decay soma +" + fmt(cf[3].mu - cf[5].mu, 4) + "."].map(s => "<li>" + s + "</li>").join("");

  /* pagina 5 */
  $("t5").textContent = "De onde vem o ganho de +" + fmt(mm - bm) + ": três degraus";
  const g1 = D.mm.adam[0] - bm, g2 = D.mm.adamw[0] - D.mm.adam[0], g3 = mm - D.mm.adamw[0];
  $("tg1").textContent = "+" + fmt(g1, 3) + " · ⅔ do ganho"; $("tg2").textContent = "+" + fmt(g2, 3); $("tg3").textContent = "+" + fmt(g3, 3) + " · ¼ do ganho";
  $("r1").innerHTML = "Trocar o lr de 1e-3 para 1e-2 e deixá-lo cair a zero com warmup + cosseno, ainda com o Adam: F1 de " + fmt(bm) + " para <b>" + fmt(D.mm.adam[0]) + "</b>. A baseline andava pouco e ficava subajustada.";
  $("r2").innerHTML = "Trocar Adam por AdamW (weight decay 0,01), com o mesmo lr e agenda: " + fmt(D.mm.adam[0]) + " para <b>" + fmt(D.mm.adamw[0]) + "</b>. Controla a norma dos pesos, mas o F1 quase não muda.";
  $("r3").innerHTML = "Usar o Muon nas 2 matrizes ocultas (AdamW no resto): " + fmt(D.mm.adamw[0]) + " para <b>" + fmt(mm) + "</b>. Só um quarto do ganho, e ele exige a agenda que decai.";

  /* pagina 6 */
  $("lD1").innerHTML = [
    "<b>Baseline:</b> ρ começa perto de 1e-3 e cai para <b>" + exp(D.rho_fim_baseline) + "</b>. A perda de treino termina em <b>" + fmt(D.perda_treino.baseline, 3) + "</b>: a rede anda pouco, está subajustada.",
    "<b>Adam lr 1e-2:</b> ρ sobe até <b>" + exp(D.rho_pico.adam) + "</b> (6× a referência) e a agenda o leva a ~0. A perda de treino termina em <b>" + fmt(D.perda_treino.adam, 3) + "</b>.",
    "<b>Duas forças:</b> lr grande no começo para andar mais; lr perto de zero no fim, porque o ruído do mini-lote faz o iterado oscilar em torno do vale e a queda o \"assenta\".",
    "O warmup de 5% mostra o ρ subindo de ~0 no início (na coluna do passo 0 a 1.600), sem pico de instabilidade."].map(s => "<li>" + s + "</li>").join("");

  /* pagina 7 */
  $("tLim").innerHTML = "Com 40 épocas o ganho é <b>+" + fmt(mm - bm) + "</b> (" + Math.floor((mm - bm) / Math.max(ms, bs)) + "× o desvio). Com 200 épocas (" + n2 + (n2 === 1 ? " semente" : " sementes") + ") cai para <b>+" + fmt(m2 - b2) + "</b>" +
    (n2 > 1 ? " (" + Math.floor((m2 - b2) / Math.max(s2, bs2)) + "× o desvio)" : "") + ". A rede ainda não saturou em 40 épocas: parte do ganho do Muon é <b>velocidade</b>. " +
    (n2 > 1 ? "Mesmo com mais treino o ganho continua maior que a variação entre sementes." : "Com 1 semente não dá para afirmar que supera o desvio.");
  const ins = D.instavel;
  $("tInst").innerHTML = "Com lr 3,0 a perda <b>sobe a " + fmt(ins.pico, 1) + "</b> no passo " + ins.passo_pico.toLocaleString("pt-BR") + " e ‖θ‖ chega a " + Math.round(ins.theta_pico) + " e continua subindo (" + Math.round(ins.theta_final) +
    " no fim): <b>passo grande demais</b>. A agenda recupera em parte (F1 " + fmt(ins.f1, 2) + "), mas a rede não volta. Ao lado, a perda do Muon desce sem picos. <br><br><b>Como ler:</b> perda subindo e ‖θ‖ crescendo sem parar é o sintoma de passo grande demais (o remédio é um lr menor). É o contrário do que o gráfico mostra para a receita vencedora.";
}

/* ---------- notas e dados (gaveta) ---------- */
const NOTAS = [
`<h3>Página 1 · roteiro de 3 minutos</h3>
<p><b>0:00–0:30 · Problema e protocolo.</b> MLP de 16 mil parâmetros no Forest Covertype (7 classes). Comparamos 5 otimizadores e 4 agendas com protocolo justo: lr ajustado por otimizador, mesmo orçamento (31.800 passos), 3 sementes na confirmação, teste usado uma vez. Nenhuma execução divergiu.</p>
<p><b>0:30–1:15 · Receita e ganho.</b> Muon lr 0,03 nas matrizes ocultas, AdamW 0,01 no resto, warmup + cosseno. Em 40 épocas: F1 0,9121 ± 0,0009 contra 0,8646 ± 0,0028, mais de 15× o desvio. No teste, 0,9104 contra 0,8689.</p>
<p><b>1:15–2:15 · Diagnóstico.</b> Dois terços do ganho vêm de lr maior com queda a zero. Aponte o gráfico: a baseline anda ρ ~5e-4 e termina subajustada (perda 0,201); com lr 1e-2 o ρ chega a 6e-3 e acaba em 0 (perda 0,132).</p>
<p><b>2:15–3:00 · Limitações.</b> Só um quarto é do Muon. Com lr constante ele perde do AdamW. Com 200 épocas o ganho cai: mostre o número de 3 sementes. O Muon custa ~9× mais por época.</p>
<p style="color:var(--ink2)">As páginas 2 a 7 são o detalhe, para perguntas.</p>`,
`<h3>Página 2 · protocolo</h3><p>As 4 etapas do enunciado, com o número de execuções de cada uma. Se perguntarem do orçamento: o enunciado só autoriza reduzir nas etapas 2–3, e usamos 40 épocas em todas por limite de CPU (Muon ~9× mais lento). Declaramos e fizemos o complemento de 200 épocas.</p>`,
`<h3>Página 3 · varredura de lr</h3><p>1 semente por ponto. O círculo grande marca o melhor de cada otimizador. Todos no interior da grade (o SGD foi estendido até 3,0). O Muon varre o lr das matrizes com a parte AdamW fixa em 0,01, o melhor do AdamW, como o enunciado manda.</p>`,
`<h3>Página 4 · agendas e confirmação</h3><p>O warmup não fez diferença aqui (cosseno puro empata). O ponto forte é a queda do lr: com lr constante o Muon perde do AdamW. A confirmação usa 3 sementes; o complemento de AdamW e Adam existe porque o top-3 foi todo Muon.</p>`,
`<h3>Página 5 · decomposição</h3><p>Cada degrau muda uma coisa: lr + agenda (Adam), depois weight decay (AdamW), depois Muon. O Adam sozinho já chega a 0,8957. Só um quarto do ganho é do Muon.</p>`,
`<h3>Página 6 · diagnóstico</h3><p>Leia a mesma coluna vertical nos três gráficos. A baseline tem ρ baixo e perda alta no fim. Com lr 1e-2 + cosseno, o ρ é maior no começo e vai a zero. A queda brusca no fim é o decaimento "assentando" no vale.</p>`,
`<h3>Página 7 · limitações</h3><p>Em 200 épocas o ganho cai. A execução instável (SGD lr 3,0) mostra "passo grande demais": perda sobe a ~4,5 e ‖θ‖ a 817.</p>
<h3>Se perguntarem</h3>
<p><b>O Muon é a causa do ganho?</b> Só de um quarto; dois terços são lr + agenda.<br>
<b>Por que o warmup não ajudou?</b> Cosseno puro empata com warmup + cosseno em 3 sementes.<br>
<b>Ajuste excessivo à validação?</b> O teste (usado uma vez) deu 0,9104 contra 0,9114 na validação.<br>
<b>Por que 40 épocas?</b> Limite de CPU, igual para todos e declarado; complemento de 200 épocas.</p>`];

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
  desenharSementes(); desenharVarredura(); desenharAgendas(); desenharDecomposicao(); desenharConfirmacao(); desenharRho("svgP1rho");
  desenharD1(); desenharLimitacoes();
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
  else if (/^[1-7]$/.test(k)) irPara(+k - 1);
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
const inicio = /^#p([1-7])$/.exec(location.hash);
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
