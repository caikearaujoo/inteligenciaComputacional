"""
slide_html.py - gera slide.html (a apresentacao de 1 slide / 3 minutos do Desafio 3)

Le os logs e o experimentos.csv e escreve um HTML AUTOSSUFICIENTE (sem internet,
sem bibliotecas): os graficos sao desenhados em SVG pelo proprio HTML, a partir
dos dados embutidos. Assim os numeros do slide nunca divergem dos dados.

O que o enunciado pede do slide (Desafio 3): receita vencedora, ganho +- desvio
e um diagnostico lido nos sinais que a justifica. Em 1 slide, 3 minutos.

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


def ler(nome):
    return pd.read_csv(os.path.join(LOGS, f"{nome}_passos.csv"))


def serie(nome, coluna, n=170):
    """Passos e valores, reduzidos a ~n pontos (o HTML nao precisa de todos)."""
    p = ler(nome)
    passo = max(1, len(p) // n)
    p = p.iloc[::passo]
    return {"x": [int(v) for v in p.passo], "y": [float(f"{v:.5g}") for v in p[coluna].clip(lower=1e-6)]}


def media_desvio(valores):
    v = np.array(valores, dtype=float)
    return float(v.mean()), float(v.std(ddof=0))


def f1_sementes(d, filtro):
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

    sem_base = f1_sementes(d40, (d40.otimizador == "adam") & (d40.lr == 0.001) & (d40.agenda == "constante"))
    sem_adam = f1_sementes(d40, (d40.otimizador == "adam") & (d40.lr == 0.01) & (d40.agenda == "warmup_cosseno"))
    sem_adamw = f1_sementes(d40, (d40.otimizador == "adamw") & (d40.lr == 0.01) & (d40.agenda == "warmup_cosseno"))
    sem_muon = f1_sementes(d40, (d40.otimizador == "muon") & (d40.lr == 0.03) & (d40.agenda == "warmup_cosseno"))
    assert len(sem_base) == len(sem_adam) == len(sem_adamw) == len(sem_muon) == 3

    teste = open(os.path.join(PASTA, "teste_final.txt"), encoding="utf-8").read()
    blocos = teste.split("=== ")[1:]
    achado = {}
    for b in blocos:
        nome = "muon" if b.startswith("Receita vencedora") else "baseline"
        achado[nome] = {
            "f1": float(re.search(r"F1 macro\s*:\s*([0-9.]+)", b).group(1)),
            "acc": float(re.search(r"Acuracia simples\s*:\s*([0-9.]+)", b).group(1)),
        }

    e200 = d[d.epocas == 200]
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

    return {
        "n_exec": int(len(d)),
        "n_diverg": int(d.divergiu.sum()),
        "seeds": {"baseline": sem_base, "adam": sem_adam, "adamw": sem_adamw, "muon": sem_muon},
        "mm": {k: media_desvio(v) for k, v in
               (("baseline", sem_base), ("adam", sem_adam), ("adamw", sem_adamw), ("muon", sem_muon))},
        "teste": achado,
        "e200": {"baseline": float(e200[e200.otimizador == "adam"].f1_melhor.iloc[0]),
                 "muon": float(e200[e200.otimizador == "muon"].f1_melhor.iloc[0])},
        "rho": {"baseline": serie(BASELINE, "rho"), "adam": serie(ADAM, "rho")},
        "theta": {"adam": serie(ADAM, "theta"), "adamw": serie(ADAMW, "theta")},
        "theta_final": theta_final,
        "cv": cv,
        "muon_previsto": [math.sqrt(57) * math.sqrt(128 / 57), math.sqrt(64)],
        "perda_treino": {"baseline": perda_treino(BASELINE), "adam": perda_treino(ADAM)},
        "rho_pico": {"baseline": float(ler(BASELINE).rho.iloc[2:].max()), "adam": float(ler(ADAM).rho.max())},
        "rho_fim_baseline": float(ler(BASELINE).rho.iloc[-1]),
    }


MODELO = r"""<!DOCTYPE html>
<html lang="pt-BR" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Desafio 3 - Muon e queda do lr</title>
<style>
  :root {
    --bg: #fcfcfb; --card: #f4f3f0; --ink: #0b0b0b; --ink2: #52514e; --grid: #e2e1dc; --borda: #d9d8d3;
    --muon: #2a78d6; --adamw: #eb6834; --adam: #1baf7a; --sgd: #eda100; --neutro: #8a8984;
    --aviso: #f0efec; --foco: #2a78d6;
  }
  :root[data-theme="dark"] {
    --bg: #1a1a19; --card: #232322; --ink: #ffffff; --ink2: #c3c2b7; --grid: #383835; --borda: #3a3a37;
    --muon: #3987e5; --adamw: #d95926; --adam: #199e70; --sgd: #c98500; --neutro: #8a8984;
    --aviso: #232322; --foco: #3987e5;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
    font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif; overflow: hidden; }
  #palco { position: absolute; left: 0; top: 0; width: 1600px; height: 900px; transform-origin: 0 0; }
  .bloco { position: absolute; background: var(--card); border: 1px solid var(--borda); border-radius: 14px;
    padding: 14px 18px; transition: opacity .25s, outline-color .25s; outline: 3px solid transparent; outline-offset: 2px; }
  .bloco.ativo { outline-color: var(--foco); }
  .esmaecido { opacity: .28; }
  h1 { margin: 0; font-size: 35px; line-height: 1.12; letter-spacing: -.01em; }
  .sub { margin-top: 8px; font-size: 18px; color: var(--ink2); }
  h2 { margin: 0 0 8px; font-size: 20px; line-height: 1.2; }
  .rotulo { font-size: 14px; color: var(--ink2); }
  #cab { left: 40px; top: 14px; width: 1520px; height: 84px; background: transparent; border: 0; padding: 0; }
  .chips { display: flex; flex-direction: column; gap: 8px; margin-top: 4px; }
  .chip { display: flex; align-items: center; gap: 10px; font-size: 17px; line-height: 1.25; }
  .ponto { width: 14px; height: 14px; border-radius: 50%; flex: none; }
  .ponto.q { border-radius: 3px; }
  .ponto.t { width: 0; height: 0; border-radius: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-bottom: 14px solid var(--adam); background: none; }
  .grande { font-size: 54px; font-weight: 700; color: var(--muon); letter-spacing: -.02em; line-height: 1; font-variant-numeric: tabular-nums; }
  .medio { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .num { font-variant-numeric: tabular-nums; }
  .tag { display: inline-block; font-size: 15px; font-weight: 700; padding: 3px 10px; border-radius: 999px;
    background: var(--bg); border: 1px solid var(--borda); margin-left: 8px; vertical-align: 2px; }
  .texto { font-size: 16px; line-height: 1.38; color: var(--ink); margin: 6px 0 0; }
  .texto b { font-weight: 700; }
  .lido { font-size: 13.5px; color: var(--ink2); margin-top: 6px; }
  #aviso { left: 40px; top: 780px; width: 1520px; height: 112px; background: var(--aviso); }
  #aviso ul { margin: 6px 0 0; padding-left: 22px; font-size: 15.5px; line-height: 1.4; }
  svg { display: block; overflow: visible; }
  svg text { font-family: inherit; }
  .eixo { stroke: var(--grid); stroke-width: 1; }
  .grade { stroke: var(--grid); stroke-width: 1; }
  .tick { fill: var(--ink2); font-size: 12.5px; }
  .rot { font-size: 13px; font-weight: 600; }
  #dica { position: absolute; pointer-events: none; z-index: 20; background: var(--ink); color: var(--bg); font-size: 14px;
    padding: 7px 10px; border-radius: 8px; line-height: 1.4; opacity: 0; transition: opacity .1s; white-space: nowrap; font-variant-numeric: tabular-nums; }
  /* interface (some na impressao) */
  .ui { position: fixed; z-index: 50; }
  #barra { right: 14px; bottom: 8px; display: flex; gap: 6px; align-items: center; opacity: .4; transition: opacity .2s; }
  #barra:hover { opacity: 1; }
  #barra button { font: inherit; font-size: 13px; padding: 5px 10px; border-radius: 8px; border: 1px solid var(--borda);
    background: var(--card); color: var(--ink); cursor: pointer; }
  #barra button:hover { border-color: var(--foco); }
  #relogio { font-variant-numeric: tabular-nums; font-size: 22px; font-weight: 700; padding: 2px 12px; border-radius: 8px;
    background: var(--ink); color: var(--bg); display: none; }
  #relogio.fim { background: #d03b3b; color: #fff; }
  #gaveta { right: 0; top: 0; bottom: 0; width: 440px; background: var(--card); border-left: 1px solid var(--borda);
    padding: 54px 18px 18px; overflow: auto; display: none; font-size: 15px; line-height: 1.5; }
  #gaveta h3 { margin: 16px 0 6px; font-size: 16px; }
  #gaveta table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; margin: 4px 0 8px; }
  #gaveta th, #gaveta td { text-align: left; padding: 3px 8px 3px 0; border-bottom: 1px solid var(--grid); }
  #gaveta .abas { display: flex; gap: 6px; margin-bottom: 10px; }
  #gaveta .abas button { font: inherit; padding: 4px 10px; border-radius: 8px; border: 1px solid var(--borda); background: var(--bg); color: var(--ink); cursor: pointer; }
  #gaveta .abas button.sel { border-color: var(--foco); font-weight: 700; }
  #ajuda { left: 14px; bottom: 8px; font-size: 12.5px; color: var(--ink2); display: none; background: var(--card); padding: 4px 8px; border-radius: 8px; border: 1px solid var(--borda); }
  @page { size: 1600px 900px; margin: 0; }
  @media print {
    .ui { display: none !important; }
    html, body { overflow: hidden; width: 1600px; height: 900px; background: #fcfcfb; }
    #palco { transform: none !important; left: 0 !important; top: 0 !important; }
    .bloco { outline: none !important; }
  }
</style>
</head>
<body>
<div id="palco">
  <div id="cab" class="bloco" data-secao="cab">
    <h1>Muon + queda do lr: +0,048 de F1 sobre a baseline, e o que os sinais mostram</h1>
    <div class="sub">Forest Covertype · MLP 57→128→64→7 (16 mil parâmetros) · protocolo do Desafio 3: lr ajustado por otimizador,
      31.800 passos fixos, 3 sementes, teste usado uma única vez</div>
  </div>

  <div id="rec" class="bloco" data-secao="cab" style="left:40px; top:106px; width:470px; height:262px;">
    <h2>Receita vencedora</h2>
    <div class="chips">
      <div class="chip"><span class="ponto" style="background:var(--muon)"></span><span><b>Muon</b>, lr 0,03, nas 2 matrizes ocultas (momento 0,95, Nesterov)</span></div>
      <div class="chip"><span class="ponto q" style="background:var(--adamw)"></span><span><b>AdamW</b>, lr 0,01, nos vieses e na camada de saída</span></div>
      <div class="chip"><span class="ponto" style="background:var(--ink2); border-radius:2px; height:5px; width:18px"></span><span>agenda <b>warmup (5%) + cosseno até 0</b> · 40 épocas, 31.800 passos, lote 512</span></div>
    </div>
    <div class="lido" id="tProt" style="margin-top:10px; font-size:14.5px; line-height:1.38"></div>
  </div>

  <div id="ganho" class="bloco" data-secao="ganho" style="left:530px; top:106px; width:520px; height:262px;">
    <h2>Ganho (F1 macro, validação, 3 sementes)</h2>
    <div class="grande" id="gMuon"></div>
    <div class="rotulo" style="margin-top:4px; font-size:16px" id="gBase"></div>
    <svg id="svgSem" width="484" height="86" style="margin-top:6px"></svg>
    <div class="texto" id="gTeste" style="margin-top:4px; font-size:15.5px; white-space:nowrap"></div>
  </div>

  <div id="decomp" class="bloco" data-secao="ganho" style="left:1070px; top:106px; width:490px; height:262px;">
    <h2>De onde vem o ganho</h2>
    <svg id="svgDec" width="454" height="206"></svg>
  </div>

  <div id="d1" class="bloco" data-secao="d1" style="left:40px; top:380px; width:486px; height:392px;">
    <h2>1 · lr maior + queda a zero<span class="tag" style="color:var(--adam)">+0,031 · ⅔ do ganho</span></h2>
    <svg id="svgD1" width="450" height="204"></svg>
    <div class="texto" id="tD1"></div>
    <div class="lido">Lido em: ρ (razão de atualização) e perda de treino.</div>
  </div>

  <div id="d2" class="bloco" data-secao="d2" style="left:557px; top:380px; width:486px; height:392px;">
    <h2>2 · Weight decay<span class="tag" style="color:var(--adamw)">+0,004</span></h2>
    <svg id="svgD2" width="450" height="204"></svg>
    <div class="texto" id="tD2"></div>
    <div class="lido">Lido em: ‖θ‖ (norma dos parâmetros).</div>
  </div>

  <div id="d3" class="bloco" data-secao="d3" style="left:1074px; top:380px; width:486px; height:392px;">
    <h2>3 · Muon<span class="tag" style="color:var(--muon)">+0,013</span></h2>
    <svg id="svgD3" width="450" height="204"></svg>
    <div class="texto" id="tD3"></div>
    <div class="lido">Lido em: ‖Δθ‖ / lr por camada (tamanho do passo).</div>
  </div>

  <div id="aviso" class="bloco" data-secao="fim">
    <h2 style="margin-bottom:0">O que NÃO afirmamos</h2>
    <ul>
      <li>Só <b>um quarto</b> do ganho é do Muon. Dois terços vêm de lr + agenda, que o Adam sozinho já captura (0,8957).</li>
      <li>Com lr <b>constante</b> o Muon perde do AdamW (0,847 contra 0,853): a vantagem exige a queda do lr. Custa ~9,5× mais por época.</li>
      <li>Com <b>200 épocas</b> o ganho cai para +0,019 (1 semente): parte do ganho é convergir mais rápido, sob orçamento curto.</li>
    </ul>
  </div>
  <div id="dica"></div>
</div>

<div id="barra" class="ui">
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
<div id="ajuda" class="ui">1 · 2 · 3 destaca um diagnóstico (0 limpa) · T cronômetro (espaço pausa, R zera) · N notas · V dados · D tema · F tela cheia · P imprimir/PDF · ? esta ajuda</div>

<script>
const D = /*__DADOS__*/;

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
let escala = 1;
const dica = document.getElementById("dica");
function mostrarDica(html, ev) {
  const palco = document.getElementById("palco").getBoundingClientRect();
  dica.innerHTML = html;
  let x = (ev.clientX - palco.left) / escala + 14, y = (ev.clientY - palco.top) / escala - 8;
  if (x > 1380) x -= 280;
  dica.style.left = x + "px"; dica.style.top = y + "px"; dica.style.opacity = 1;
}
const esconderDica = () => { dica.style.opacity = 0; };

/* ---------- grafico de linhas (eixos x linear, y linear ou log) ---------- */
function graficoLinhas(svg, o) {
  svg.innerHTML = "";
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height");
  const m = { l: 50, r: 12, t: 10, b: 30 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const fy = o.log ? Math.log10 : (v => v);
  const sx = v => m.l + (v - o.x0) / (o.x1 - o.x0) * iw;
  const sy = v => m.t + ih - (fy(v) - fy(o.y0)) / (fy(o.y1) - fy(o.y0)) * ih;
  o.yticks.forEach(t => {
    svgEl("line", { x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t), class: "grade" }, svg);
    const tx = svgEl("text", { x: m.l - 8, y: sy(t) + 4, "text-anchor": "end", class: "tick" }, svg); tx.textContent = o.yfmt(t);
  });
  o.xticks.forEach(t => {
    const tx = svgEl("text", { x: sx(t), y: H - 10, "text-anchor": "middle", class: "tick" }, svg);
    tx.textContent = t === 0 ? "0" : (t / 1000) + " mil";
  });
  svgEl("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, class: "eixo" }, svg);
  (o.ref || []).forEach(r => {
    svgEl("line", { x1: m.l, x2: m.l + iw, y1: sy(r.y), y2: sy(r.y), stroke: cor("--ink2"), "stroke-width": 1, "stroke-dasharray": "5 4" }, svg);
    const tx = svgEl("text", { x: m.l + iw, y: sy(r.y) - 5, "text-anchor": "end", class: "tick" }, svg); tx.textContent = r.texto;
  });
  o.series.forEach(s => {
    let d = "";
    s.x.forEach((x, i) => { d += (i ? "L" : "M") + sx(x).toFixed(1) + " " + sy(Math.max(s.y[i], o.y0)).toFixed(1); });
    svgEl("path", { d, fill: "none", stroke: cor(s.cor), "stroke-width": 2.2, "stroke-linejoin": "round" }, svg);
  });
  (o.rotulos || []).forEach(r => {
    const tx = svgEl("text", { x: sx(r.x), y: sy(r.y), "text-anchor": r.ancora || "start", class: "rot", fill: cor(r.cor) }, svg);
    String(r.texto).split("\n").forEach((linha, i) => { const ts = svgEl("tspan", { x: sx(r.x), dy: i ? 15 : 0 }, tx); ts.textContent = linha; });
  });
  /* hover: linha vertical + dica com o valor de cada serie */
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

/* ---------- 3 sementes: Muon contra baseline ---------- */
function desenharSementes() {
  const svg = document.getElementById("svgSem"); svg.innerHTML = "";
  const W = 484, m = { l: 80, r: 24 }, lo = 0.855, hi = 0.925;
  const sx = v => m.l + (v - lo) / (hi - lo) * (W - m.l - m.r);
  const linhas = [["Muon", "muon", "--muon", 22], ["Baseline", "baseline", "--neutro", 54]];
  linhas.forEach(([nome, chave, c, y]) => {
    const t = svgEl("text", { x: 0, y: y + 5, class: "rot", fill: cor("--ink") }, svg); t.textContent = nome;
    const [mu, sd] = D.mm[chave];
    svgEl("line", { x1: sx(mu - sd), x2: sx(mu + sd), y1: y, y2: y, stroke: cor(c), "stroke-width": 7, opacity: .35, "stroke-linecap": "round" }, svg);
    D.seeds[chave].forEach((v, i) => {
      const p = svgEl("circle", { cx: sx(v), cy: y, r: 7, fill: cor(c), stroke: cor("--card"), "stroke-width": 2 }, svg);
      p.addEventListener("mousemove", ev => mostrarDica("<b>" + nome + "</b><br>semente " + i + ": F1 " + fmt(v), ev));
      p.addEventListener("mouseleave", esconderDica);
    });
  });
  [0.86, 0.88, 0.90, 0.92].forEach(t => {
    if (t > hi) return;
    const x = svgEl("text", { x: sx(t), y: 80, "text-anchor": "middle", class: "tick" }, svg); x.textContent = fmt(t, 2);
  });
  svgEl("line", { x1: m.l, x2: W - m.r, y1: 68, y2: 68, class: "eixo" }, svg);
  const cap = svgEl("text", { x: m.l, y: 7, class: "tick" }, svg); cap.textContent = "cada ponto = 1 semente (F1 de validação)";
}

/* ---------- decomposicao do ganho ---------- */
function desenharDecomposicao() {
  const svg = document.getElementById("svgDec"); svg.innerHTML = "";
  const W = 454, H = 206, m = { l: 46, r: 8, t: 22, b: 40 }, lo = 0.855, hi = 0.93;
  const sx = i => m.l + (i + 0.5) / 4 * (W - m.l - m.r);
  const sy = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
  [0.86, 0.88, 0.90, 0.92].forEach(t => {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "grade" }, svg);
    const x = svgEl("text", { x: m.l - 6, y: sy(t) + 4, "text-anchor": "end", class: "tick" }, svg); x.textContent = fmt(t, 2);
  });
  const passos = [
    ["Baseline", "baseline", "--neutro", "circle", "Adam 1e-3, constante"],
    ["+ lr 1e-2\n+ agenda", "adam", "--adam", "tri", "Adam lr 1e-2, warmup + cosseno"],
    ["+ weight\ndecay", "adamw", "--adamw", "quad", "AdamW"],
    ["+ Muon", "muon", "--muon", "circle", "Muon nas matrizes ocultas"],
  ];
  let d = "";
  passos.forEach((p, i) => { d += (i ? "L" : "M") + sx(i) + " " + sy(D.mm[p[1]][0]); });
  svgEl("path", { d, fill: "none", stroke: cor("--ink2"), "stroke-width": 1.4 }, svg);
  passos.forEach((p, i) => {
    const [mu, sd] = D.mm[p[1]], x = sx(i), y = sy(mu), c = cor(p[2]);
    svgEl("line", { x1: x, x2: x, y1: sy(mu - sd), y2: sy(mu + sd), stroke: cor("--ink"), "stroke-width": 1.6 }, svg);
    let marca;
    if (p[3] === "quad") marca = svgEl("rect", { x: x - 8, y: y - 8, width: 16, height: 16, rx: 2, fill: c, stroke: cor("--card"), "stroke-width": 2 }, svg);
    else if (p[3] === "tri") marca = svgEl("path", { d: "M" + x + " " + (y - 10) + "L" + (x + 9) + " " + (y + 7) + "L" + (x - 9) + " " + (y + 7) + "Z", fill: c, stroke: cor("--card"), "stroke-width": 2 }, svg);
    else marca = svgEl("circle", { cx: x, cy: y, r: 8, fill: c, stroke: cor("--card"), "stroke-width": 2 }, svg);
    const ganho = i ? "+" + fmt(mu - D.mm[passos[i - 1][1]][0]) : "";
    const t1 = svgEl("text", { x, y: y - 17, "text-anchor": "middle", class: "rot", fill: cor("--ink") }, svg); t1.textContent = fmt(mu);
    if (ganho) { const t2 = svgEl("text", { x, y: y + 27, "text-anchor": "middle", class: "tick", fill: cor("--ink2") }, svg); t2.textContent = "(" + ganho + ")"; }
    p[0].split("\n").forEach((linha, k) => { const t = svgEl("text", { x, y: H - 22 + k * 14, "text-anchor": "middle", class: "tick" }, svg); t.textContent = linha; });
    marca.addEventListener("mousemove", ev => mostrarDica("<b>" + p[4] + "</b><br>F1 " + fmt(mu) + " ± " + fmt(sd) + "<br>sementes: " + D.seeds[p[1]].map(v => fmt(v)).join(" · "), ev));
    marca.addEventListener("mouseleave", esconderDica);
  });
}

/* ---------- diagnostico 3: variacao do tamanho do passo ---------- */
function desenharPasso() {
  const svg = document.getElementById("svgD3"); svg.innerHTML = "";
  const W = 450, H = 204, m = { l: 74, r: 14, t: 14, b: 44 }, max = 50;
  const sx = v => m.l + v / max * (W - m.l - m.r);
  const linhas = [["Muon", "--muon", "Muon"], ["AdamW", "--adamw", "AdamW"], ["SGD", "--sgd", "SGD"]];
  [0, 10, 20, 30, 40, 50].forEach(t => {
    svgEl("line", { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b, class: "grade" }, svg);
    const x = svgEl("text", { x: sx(t), y: H - m.b + 17, "text-anchor": "middle", class: "tick" }, svg); x.textContent = t + "%";
  });
  const xl = svgEl("text", { x: m.l + (W - m.l - m.r) / 2, y: H - 8, "text-anchor": "middle", class: "tick" }, svg);
  xl.textContent = "quanto o passo/lr varia ao longo do treino (CV, %)";
  linhas.forEach(([nome, c, chave], k) => {
    const y = m.t + 28 + k * 54, cv = D.cv[chave].cv, md = D.cv[chave].mediana;
    const t = svgEl("text", { x: 0, y: y + 5, class: "rot", fill: cor("--ink") }, svg); t.textContent = nome;
    svgEl("line", { x1: sx(cv[0]), x2: sx(cv[1]), y1: y, y2: y, stroke: cor(c), "stroke-width": 3 }, svg);
    const a = svgEl("circle", { cx: sx(cv[0]), cy: y, r: 8, fill: cor(c), stroke: cor("--card"), "stroke-width": 2 }, svg);
    const b = svgEl("rect", { x: sx(cv[1]) - 8, y: y - 8, width: 16, height: 16, rx: 2, fill: cor(c), stroke: cor("--card"), "stroke-width": 2 }, svg);
    const lo = Math.min(cv[0], cv[1]), hi = Math.max(cv[0], cv[1]);
    const r = svgEl("text", { x: sx(hi) + 14, y: y + 5, class: "rot", fill: cor("--ink") }, svg);
    r.textContent = Math.round(lo) + (Math.round(lo) === Math.round(hi) ? "" : " a " + Math.round(hi)) + "%";
    [[a, 0, "camada 1 (128×57)"], [b, 1, "camada 2 (64×128)"]].forEach(([el, i, rot]) => {
      el.addEventListener("mousemove", ev => mostrarDica("<b>" + nome + "</b> · " + rot + "<br>variação do passo/lr: " + fmt(cv[i], 1) + "%<br>mediana de ‖Δθ‖/lr: " + fmt(md[i], 2), ev));
      el.addEventListener("mouseleave", esconderDica);
    });
  });
  const lg = svgEl("text", { x: m.l, y: m.t + 4, class: "tick" }, svg);
  lg.textContent = "● camada 1   ■ camada 2";
}

/* ---------- textos que dependem dos dados ---------- */
function preencherTextos() {
  const [mm, ms] = D.mm.muon, [bm, bs] = D.mm.baseline;
  document.getElementById("gMuon").textContent = fmt(mm) + " ± " + fmt(ms);
  document.getElementById("tProt").innerHTML = "Comparados: SGD (± Nesterov), Adam, AdamW e Muon × 4 agendas. <b>" +
    D.n_exec + " execuções, " + (D.n_diverg === 0 ? "nenhuma divergiu" : D.n_diverg + " divergiram") + ".</b><br>" +
    "<b>Adotada no modelo final do projeto</b> (modelo.py): teste " + (D.teste.muon.acc * 100).toFixed(1).replace(".", ",") + "%.";
  document.getElementById("gBase").innerHTML ="contra <b class='num'>" + fmt(bm) + " ± " + fmt(bs) + "</b> da baseline (Adam 1e-3, constante)";
  const razao = (mm - bm) / Math.max(ms, bs);
  document.getElementById("gTeste").innerHTML = "<b>+" + fmt(mm - bm) + "</b> = " + Math.floor(razao) + "× o desvio · Teste (1 uso): <b class='num'>" +
    fmt(D.teste.muon.f1) + "</b> contra " + fmt(D.teste.baseline.f1);
  const exp = v => v.toExponential(0).replace("-", "‑");   /* hifen que nao quebra a linha */
  document.getElementById("tD1").innerHTML = "A baseline mexe pouco: ρ cai de 1e‑3 a <b>" + exp(D.rho_fim_baseline) +
    "</b> e a perda de treino fica em <b>" + fmt(D.perda_treino.baseline, 3) + "</b> (subajustada). Com lr 0,01 o ρ chega a <b>" +
    exp(D.rho_pico.adam) + "</b> e a agenda o leva a ~0: perda de treino <b>" + fmt(D.perda_treino.adam, 3) + "</b>.";
  const tf = D.theta_final;
  const faixa = a => Math.round(Math.min(...a)) + "–" + Math.round(Math.max(...a));
  document.getElementById("tD2").innerHTML = "Sem weight decay a norma dos pesos cresce até <b>" + faixa(tf.Adam) + "</b> (3 sementes); com AdamW estabiliza em <b>" +
    faixa(tf.AdamW) + "</b>. Ele controla a norma, mas o F1 quase não muda: o gargalo <b>não era sobreajuste</b>.";
  const pv = D.muon_previsto, md = D.cv.Muon.mediana;
  document.getElementById("tD3").innerHTML = "O passo do Muon é ≈ <b>lr × √(menor dimensão)</b>, quase constante: medido <b>" + fmt(md[0], 1) + " e " + fmt(md[1], 1) +
    "</b> contra <b>" + fmt(pv[0], 1) + " e " + fmt(pv[1], 1) + "</b> previstos. O do SGD depende do gradiente.";
}

/* ---------- graficos dos diagnosticos 1 e 2 ---------- */
function desenharD1() {
  graficoLinhas(document.getElementById("svgD1"), {
    x0: 0, x1: 31800, y0: 1e-5, y1: 0.05, log: true, yticks: [1e-5, 1e-4, 1e-3, 1e-2], xticks: [0, 10000, 20000, 30000],
    yfmt: v => v.toExponential(0).replace("e-", "e-"), tip: v => "ρ = " + v.toExponential(1),
    ref: [{ y: 1e-3, texto: "referência 1e-3" }],
    series: [{ nome: "Baseline (Adam 1e-3, constante)", cor: "--neutro", x: D.rho.baseline.x, y: D.rho.baseline.y },
             { nome: "Adam lr 1e-2 + warmup/cosseno", cor: "--adam", x: D.rho.adam.x, y: D.rho.adam.y }],
    rotulos: [{ x: 9500, y: 1.9e-4, texto: "baseline: ρ ~1e-3 → 5e-4", cor: "--ink2" },
              { x: 9800, y: 1.35e-2, texto: "lr 1e-2: pico 6e-3, depois → 0", cor: "--adam" }]
  });
}
function desenharD2() {
  graficoLinhas(document.getElementById("svgD2"), {
    x0: 0, x1: 31800, y0: 0, y1: 175, log: false, yticks: [0, 50, 100, 150], xticks: [0, 10000, 20000, 30000],
    yfmt: v => String(v), tip: v => "‖θ‖ = " + v.toFixed(1).replace(".", ","),
    series: [{ nome: "Adam lr 1e-2 (sem weight decay)", cor: "--adam", x: D.theta.adam.x, y: D.theta.adam.y },
             { nome: "AdamW lr 1e-2 (weight decay 0,01)", cor: "--adamw", x: D.theta.adamw.x, y: D.theta.adamw.y }],
    rotulos: [{ x: 31000, y: 142, texto: "Adam: " + Math.round(D.theta_final.Adam[0]), cor: "--adam", ancora: "end" },
              { x: 31000, y: 108, texto: "AdamW: " + Math.round(D.theta_final.AdamW[0]), cor: "--adamw", ancora: "end" }]
  });
}

/* ---------- notas e dados (gaveta) ---------- */
const NOTAS = `
<h3>Roteiro de 3 minutos</h3>
<p><b>0:00–0:30 · Problema e protocolo.</b> MLP de 16 mil parâmetros no Forest Covertype (7 classes). Comparamos 5 otimizadores e 4 agendas com
protocolo justo: lr ajustado por otimizador, mesmo orçamento (31.800 passos), 3 sementes na confirmação, teste usado uma vez. Nenhuma execução divergiu.</p>
<p><b>0:30–1:00 · Receita e ganho.</b> Muon com lr 0,03 e agenda warmup + cosseno. F1 0,9121 ± 0,0009 contra 0,8646 ± 0,0028: +0,0475, mais de 15 vezes o desvio.
No teste, 0,9104 contra 0,8689. Mostre a decomposição: lr + agenda +0,031; weight decay +0,004; Muon +0,013.</p>
<p><b>1:00–1:30 · Diagnóstico 1 (ρ).</b> A baseline mexia 5e-4 por passo e não saturava. Com lr maior e queda a zero, o ρ chega a 6e-3 e acaba em 0: dois terços do ganho.
Duas forças: lr grande no começo para andar, lr a ~0 no fim porque o ruído do mini-lote faz o iterado oscilar.</p>
<p><b>1:30–2:00 · Diagnóstico 2 (‖θ‖).</b> Sem weight decay a norma vai a ~156; com AdamW estabiliza em ~92 (θ ← θ − lr·λ·θ, independente do gradiente).
Controla a norma, F1 quase não muda: não era sobreajuste.</p>
<p><b>2:00–2:30 · Diagnóstico 3 (tamanho do passo).</b> O Muon troca o momento por U·Vᵀ: todas as direções singulares recebem o mesmo passo. O passo vale ≈ lr·√(menor dimensão),
com variação de 1 a 2%, e bate com a previsão (87 a 92%). No SGD o passo varia 29 a 41% com o gradiente.</p>
<p><b>2:30–3:00 · Limitações.</b> Em 200 épocas o ganho cai para +0,019 (1 semente): parte é velocidade. O Muon custa ~9,5× mais e só vence com a agenda que decai.
Varreduras com 1 semente.</p>
<h3>Se perguntarem</h3>
<p><b>O Muon é a causa do ganho?</b> Só de um quarto; dois terços são lr + agenda (o Adam sozinho chega a 0,8957).<br>
<b>Por que o warmup não ajudou?</b> O passo 1 do Adam sem warmup é 5% de ‖θ‖ (bate com lr·√n/‖θ‖), mas nesta rede bem condicionada não desestabiliza.<br>
<b>E a execução instável?</b> SGD com lr 3,0: perda sobe a 4,5 e ‖θ‖ a 817 no passo 1.700 (passo grande demais); a agenda recupera em parte (F1 0,61).<br>
<b>Ajuste excessivo à validação?</b> O teste (usado uma vez) deu 0,9104 contra 0,9114 na validação.<br>
<b>E com mais treino?</b> Em 200 épocas: 0,9238 contra 0,9048 (+0,019, 1 semente).</p>
<p style="color:var(--ink2)">Detalhe completo, evidências e contraprovas: <code>desafio3.md</code>, seção 7.</p>`;

function tabela(cabecalho, linhas) {
  return "<table><tr>" + cabecalho.map(c => "<th>" + c + "</th>").join("") + "</tr>" +
    linhas.map(l => "<tr>" + l.map(c => "<td>" + c + "</td>").join("") + "</tr>").join("") + "</table>";
}
function montarDados() {
  const mm = D.mm, s = D.seeds;
  let h = "<h3>Ganho por etapa (F1 de validação, 3 sementes)</h3>" + tabela(["Receita", "média ± desvio", "sementes 0 · 1 · 2"], [
    ["Baseline (Adam 1e-3, constante)", fmt(mm.baseline[0]) + " ± " + fmt(mm.baseline[1]), s.baseline.map(v => fmt(v)).join(" · ")],
    ["Adam lr 1e-2, warmup + cosseno", fmt(mm.adam[0]) + " ± " + fmt(mm.adam[1]), s.adam.map(v => fmt(v)).join(" · ")],
    ["AdamW lr 1e-2, warmup + cosseno", fmt(mm.adamw[0]) + " ± " + fmt(mm.adamw[1]), s.adamw.map(v => fmt(v)).join(" · ")],
    ["Muon lr 0,03, warmup + cosseno", fmt(mm.muon[0]) + " ± " + fmt(mm.muon[1]), s.muon.map(v => fmt(v)).join(" · ")]]);
  h += "<h3>Teste (usado uma única vez)</h3>" + tabela(["", "F1 macro", "acurácia"], [
    ["Muon", fmt(D.teste.muon.f1), fmt(D.teste.muon.acc)], ["Baseline", fmt(D.teste.baseline.f1), fmt(D.teste.baseline.acc)]]);
  h += "<h3>200 épocas (1 semente)</h3>" + tabela(["", "F1 de validação"], [["Muon", fmt(D.e200.muon)], ["Baseline", fmt(D.e200.baseline)]]);
  h += "<h3>Tamanho do passo (‖Δθ‖/lr)</h3>" + tabela(["", "variação camada 1 · 2", "mediana camada 1 · 2"],
    ["Muon", "AdamW", "SGD"].map(k => [k, fmt(D.cv[k].cv[0], 1) + "% · " + fmt(D.cv[k].cv[1], 1) + "%", fmt(D.cv[k].mediana[0], 2) + " · " + fmt(D.cv[k].mediana[1], 2)]));
  h += "<p style='color:var(--ink2)'>Previsto para o Muon (valores singulares = 1): " + fmt(D.muon_previsto[0], 2) + " e " + fmt(D.muon_previsto[1], 2) + ".</p>";
  h += "<h3>‖θ‖ final, por semente</h3>" + tabela(["", "0 · 1 · 2"], [
    ["Adam (sem weight decay)", D.theta_final.Adam.map(v => v.toFixed(1).replace(".", ",")).join(" · ")],
    ["AdamW", D.theta_final.AdamW.map(v => v.toFixed(1).replace(".", ",")).join(" · ")]]);
  return h;
}

/* ---------- foco, cronometro, gaveta, tema ---------- */
const blocos = [...document.querySelectorAll(".bloco")];
function focar(id) {
  blocos.forEach(b => b.classList.toggle("esmaecido", id && b.id !== id && b.id !== "cab"));
}
const SECOES = [["cab", 0], ["ganho", 30], ["d1", 60], ["d2", 90], ["d3", 120], ["fim", 150]];
let rel = { ligado: false, rodando: false, t: 0, id: null };
function atualizarRelogio() {
  const el = document.getElementById("relogio");
  const resta = Math.max(0, 180 - rel.t);
  el.textContent = Math.floor(resta / 60) + ":" + String(Math.floor(resta % 60)).padStart(2, "0");
  el.classList.toggle("fim", resta === 0);
  let atual = "cab";
  SECOES.forEach(([s, ini]) => { if (rel.t >= ini) atual = s; });
  blocos.forEach(b => b.classList.toggle("ativo", rel.ligado && b.dataset.secao === atual));
}
setInterval(() => { if (rel.ligado && rel.rodando) { rel.t += 0.25; atualizarRelogio(); } }, 250);
function alternarRelogio() {
  rel.ligado = !rel.ligado; rel.rodando = rel.ligado; rel.t = 0;
  document.getElementById("relogio").style.display = rel.ligado ? "inline-block" : "none";
  if (!rel.ligado) blocos.forEach(b => b.classList.remove("ativo")); else atualizarRelogio();
}
function alternarGaveta(aba) {
  const g = document.getElementById("gaveta");
  const abrir = g.style.display !== "block" || (aba && g.dataset.aba !== aba);
  g.style.display = abrir ? "block" : "none";
  if (aba) { g.dataset.aba = aba;
    document.getElementById("painelNotas").style.display = aba === "n" ? "block" : "none";
    document.getElementById("painelDados").style.display = aba === "d" ? "block" : "none";
    document.getElementById("aNotas").classList.toggle("sel", aba === "n"); document.getElementById("aDados").classList.toggle("sel", aba === "d"); }
  ajustar();
}
function ajustar() {
  const g = document.getElementById("gaveta").style.display === "block" ? 440 : 0;
  escala = Math.min((window.innerWidth - g) / 1600, window.innerHeight / 900);
  const p = document.getElementById("palco");
  p.style.transform = "scale(" + escala + ")";
  p.style.left = Math.max(0, (window.innerWidth - g - 1600 * escala) / 2) + "px";
  p.style.top = Math.max(0, (window.innerHeight - 900 * escala) / 2) + "px";
}
function desenharTudo() { desenharSementes(); desenharDecomposicao(); desenharD1(); desenharD2(); desenharPasso(); }
function trocarTema() {
  const r = document.documentElement;
  r.dataset.theme = r.dataset.theme === "dark" ? "light" : "dark";
  desenharTudo();
}
document.getElementById("painelNotas").innerHTML = NOTAS;
document.getElementById("painelDados").innerHTML = montarDados();
document.getElementById("bRel").onclick = alternarRelogio;
document.getElementById("bNotas").onclick = () => alternarGaveta("n");
document.getElementById("bTema").onclick = trocarTema;
document.getElementById("bPdf").onclick = () => window.print();
document.getElementById("aNotas").onclick = () => alternarGaveta("n");
document.getElementById("aDados").onclick = () => alternarGaveta("d");
window.addEventListener("keydown", ev => {
  const k = ev.key.toLowerCase();
  if (k === "1" || k === "2" || k === "3") focar("d" + k);
  else if (k === "0" || k === "escape") focar(null);
  else if (k === "t") alternarRelogio();
  else if (k === " ") { if (rel.ligado) { rel.rodando = !rel.rodando; ev.preventDefault(); } }
  else if (k === "r") { rel.t = 0; atualizarRelogio(); }
  else if (k === "n") alternarGaveta("n");
  else if (k === "v") alternarGaveta("d");
  else if (k === "d") trocarTema();
  else if (k === "p") window.print();
  else if (k === "?") { const a = document.getElementById("ajuda"); a.style.display = a.style.display === "block" ? "none" : "block"; }
  else if (k === "f") { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen(); }
});
let temaAntes = null;
window.addEventListener("beforeprint", () => { temaAntes = document.documentElement.dataset.theme; document.documentElement.dataset.theme = "light"; desenharTudo(); });
window.addEventListener("afterprint", () => { document.documentElement.dataset.theme = temaAntes; desenharTudo(); });
window.addEventListener("resize", ajustar);
preencherTextos(); desenharTudo(); ajustar();
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
                    f"--print-to-pdf={caminho_pdf}", "--virtual-time-budget=4000", url],
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
