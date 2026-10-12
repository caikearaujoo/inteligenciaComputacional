#!/bin/bash
# Etapa 3: as 4 agendas para os 2 melhores otimizadores da etapa 2 (Muon e AdamW),
# cada um com o seu melhor lr. warmup_cosseno ja foi rodada na etapa 2.
# AdamW lr 0.01; Muon lr 0.03 com a parte AdamW em 0.01. Semente 0, 40 epocas.
cd "$(dirname "$0")"
for ag in constante cosseno wsd; do
  py -u treino.py --otimizador adamw --lr 0.01 --agenda $ag --semente 0 --epocas 40
done
for ag in constante cosseno wsd; do
  py -u treino.py --otimizador muon --lr 0.03 --lr_adamw 0.01 --agenda $ag --semente 0 --epocas 40
done
