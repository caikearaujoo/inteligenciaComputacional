"""
desafio2_CaikeCesarMotaDeAraujoMatheusIzidroCamposDosSantos.py
Desafio 2 (GBC073) — Ativação e inicialização em redes profundas
Alunos: Caike Cesar Mota de Araujo, Matheus Izidro Campos dos Santos

A conta que resolve o desafio: se z ~ N(0, 1) e W ~ N(0, s^2), a variância da
pré-ativação da camada seguinte é

    Var(z') = fan_in * s^2 * E[f(z)^2]

Para manter essa variância em 1 ao longo de L camadas (nem sumir, nem
explodir):

    s^2 = 1 / (fan_in * E[f(z)^2])

Testamos SELU e GELU antes de chegar aqui, e as duas falharam em profundidade
mesmo com a variância calibrada certinha no passo 0 (raio-X mostrando ~1 em
todas as camadas). O motivo: a fórmula assume que a entrada de cada camada é
z ~ N(0, 1) i.i.d., mas dados reais (imagens) têm pixels correlacionados —
já na 1ª camada a variância medida saía em ~2,4, não em 1. Para uma ativação
NÃO homogênea como o GELU (f(c·z) != c·f(z) para c != 1), esse desvio inicial
se acumula multiplicativamente a cada camada e explode (chegou a ~3600 na
camada 48). ReLU não tem esse problema: é positivamente homogênea —
ReLU(c·z) = c·ReLU(z) para c > 0 — então, mesmo que a variância de entrada
já esteja errada em relação ao ideal N(0,1), o fator de ganho por camada
continua correto e o desvio NÃO se acumula com a profundidade.

Estimamos E[f(z)^2] por amostragem (Monte Carlo) uma única vez, no import do
módulo, em vez de usar 0.5 direto de cabeça — o mesmo código funciona pra
qualquer ativação elemento a elemento, só troca a linha de `ativacao`.
"""
import math
import torch
import torch.nn.functional as F


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)  # elemento a elemento, sem parâmetros, positivamente homogênea


# E[f(z)^2] com z ~ N(0,1), estimado uma vez por amostragem (Monte Carlo)
_gerador = torch.Generator().manual_seed(0)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_gerador)).pow(2).mean().item()


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                 fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    desvio = math.sqrt(1.0 / (fan_in * _E_f2))
    if camada == n_camadas:
        # camada de logits: reduzir a escala evita que o SGD comece com
        # passos grandes demais na saída da rede
        desvio *= 0.5
    W.normal_(0.0, desvio)
    b.zero_()
