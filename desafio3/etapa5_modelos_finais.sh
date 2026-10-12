#!/bin/bash
# Retreina, com --salvar, a receita vencedora e a baseline (semente 0, 40 epocas) para o teste final.
# Servem tambem de verificacao de reprodutibilidade: o F1 de validacao tem que repetir
# 0.9114 (Muon) e 0.8683 (baseline), os valores ja registrados na etapa 2/baseline.
cd "$(dirname "$0")"
py -u treino.py --otimizador adam --lr 0.001 --agenda constante --semente 0 --epocas 40 --salvar --nome final_baseline_s0_e40
py -u treino.py --otimizador muon --lr 0.03 --lr_adamw 0.01 --agenda warmup_cosseno --semente 0 --epocas 40 --salvar --nome final_muon_s0_e40
