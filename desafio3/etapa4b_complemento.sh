#!/bin/bash
# Complemento da etapa 4 (fora do top-3 do protocolo, declarado como tal):
# sementes 1 e 2 do AdamW (cosseno e warmup_cosseno, lr 0.01) e do Adam (warmup_cosseno, lr 0.01).
# Motivo: o top-3 da etapa 3 foi todo Muon, entao "Muon > AdamW" ficaria sustentado por 1 semente
# (mesma agenda: cosseno), e "AdamW > Adam" (efeito do weight decay) tambem.
# Espera o driver da etapa 4 terminar para nao competir por CPU (os tempos ficam limpos).
cd "$(dirname "$0")"
until grep -q "ETAPA 4 COMPLETA" logs_etapa4.txt; do sleep 30; done
for s in 1 2; do
  py -u treino.py --otimizador adamw --lr 0.01 --agenda cosseno --semente $s --epocas 40
  py -u treino.py --otimizador adamw --lr 0.01 --agenda warmup_cosseno --semente $s --epocas 40
  py -u treino.py --otimizador adam --lr 0.01 --agenda warmup_cosseno --semente $s --epocas 40
done
echo "COMPLEMENTO COMPLETO"
