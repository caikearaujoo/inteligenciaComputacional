#!/bin/bash
# Etapa 2, parte 2. Mesmas condicoes da parte 1 (warmup + cosseno, semente 0, 40 epocas).
# (a) SGD e SGD Nesterov: o melhor lr da grade (0.3) caiu na borda, entao a grade e estendida.
# (b) Muon: varre o lr das matrizes; a parte AdamW fica no melhor lr do AdamW (0.01).
cd "$(dirname "$0")"
for lr in 1.0 3.0; do
  for o in sgd sgd_nesterov; do
    py -u treino.py --otimizador $o --lr $lr --agenda warmup_cosseno --semente 0 --epocas 40
  done
done
for lr in 0.003 0.01 0.03 0.1 0.3; do
  py -u treino.py --otimizador muon --lr $lr --lr_adamw 0.01 --agenda warmup_cosseno --semente 0 --epocas 40
done
