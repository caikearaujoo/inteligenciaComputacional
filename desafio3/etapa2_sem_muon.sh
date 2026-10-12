#!/bin/bash
# Etapa 2 do protocolo (varredura de lr), sem o Muon.
# warmup + cosseno, 1 semente (0), 40 epocas, lote 512 (igual para todos).
cd "$(dirname "$0")"
for lr in 0.003 0.01 0.03 0.1 0.3; do
  for o in sgd sgd_nesterov; do
    py -u treino.py --otimizador $o --lr $lr --agenda warmup_cosseno --semente 0 --epocas 40
  done
done
for lr in 0.0003 0.001 0.003 0.01 0.03; do
  for o in adam adamw; do
    py -u treino.py --otimizador $o --lr $lr --agenda warmup_cosseno --semente 0 --epocas 40
  done
done
