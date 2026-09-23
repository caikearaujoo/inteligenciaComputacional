"""
gerar_slide.py — gera desafio2/slide.pdf a partir de dados REAIS do raio-X
(nada de curva inventada: os 48 pontos vêm de uma chamada direta a
harness_desafio2.rodar() com nossa submissão, task mnist_L48, semente 0).

Não faz parte da entrega do Desafio 2 em si (o harness não pede isso) — é só
a ferramenta usada uma vez para montar o slide.pdf.
"""
import sys
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def carregar_modulo(nome, caminho):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


harness = carregar_modulo("harness_desafio2", "harness_desafio2.py")
sub = carregar_modulo("submissao", "desafio2_CaikeCesarMotaDeAraujoMatheusIzidroCamposDosSantos.py")

# raio-X real (passo 0) da nossa submissão, MNIST, L=48 -- a tarefa mais dura
res = harness.rodar(sub, "mnist", 48, 256, semente=0, fracao=1.0, epocas=0)
variancias = res.var_pre_ativ  # 48 valores reais, um por camada

fig = plt.figure(figsize=(13.33, 8.4))  # 16:9-ish, um pouco mais alto pra caber o texto sem sobrepor
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], left=0.045, right=0.975, top=0.93, bottom=0.09, wspace=0.28)

# ---------------- painel de texto (esquerda) ----------------
ax_txt = fig.add_subplot(gs[0, 0])
ax_txt.axis("off")

linhas = [
    r"$\bf{Desafio\ 2\ -\ Ativação\ e\ Inicialização}$",
    "",
    r"$\bf{O\ que\ escolhemos:}$",
    r"  Ativação: $\mathrm{ReLU}(x) = \max(0, x)$  —  zera os",
    "  valores negativos, deixa os positivos passar.",
    "",
    r"  Peso inicial: em vez de usar uma fórmula pronta,",
    "  calculamos o tamanho certo dos pesos simulando a rede",
    "  1 milhão de vezes antes do treino começar.",
    "",
    r"$\bf{Por\ que\ isso\ importa?}$",
    "  Numa rede de 48 camadas, um pequeno erro na escala dos",
    "  pesos se multiplica a cada camada. Testamos 3 ativações:",
    "  • SELU: parecia perfeita no papel, mas não treinava.",
    "  • GELU: o sinal foi explodindo a cada camada (efeito",
    "    \"telefone sem fio\") até virar um número gigante.",
    "  • ReLU: o sinal se manteve estável — foi a que funcionou.",
    "",
    r"$\bf{Resultado:}$",
    "  Nota final: 55,2 de 100.",
    "  Nota máxima em 2 das 9 tarefas testadas, e dentro da faixa",
    "  esperada em mais da metade delas no total.",
    "  Nas 2 tarefas com redes mais profundas (48 camadas em",
    "  MNIST/Fashion), nenhuma configuração testada aprendeu —",
    "  a acurácia ficou no nível de chute aleatório.",
]
texto = "\n".join(linhas)

ax_txt.text(0.0, 1.0, texto, transform=ax_txt.transAxes, fontsize=13,
            va="top", ha="left", linespacing=1.65, family="DejaVu Sans")

# ---------------- gráfico (direita): dado real, sem invenção ----------------
ax = fig.add_subplot(gs[0, 1])
camadas = list(range(1, len(variancias) + 1))
ax.plot(camadas, variancias, "-o", color="#3f6dc4", markersize=3.5, linewidth=1.6,
        label="O que medimos, camada por camada")
ax.axhline(1.0, color="#e9a100", linestyle="--", linewidth=1.6, label="O ideal seria ficar aqui")
ax.set_xlabel("Camada da rede (1 a 48)")
ax.set_ylabel("Força do sinal")
ax.set_title("O sinal se mantém sob controle ao longo das 48 camadas")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)
ax.annotate("baixamos de propósito\nna última camada",
            xy=(camadas[-1], variancias[-1]), xytext=(camadas[-1] - 22, variancias[-1] + 2.3),
            fontsize=8.5, color="#55607a",
            arrowprops=dict(arrowstyle="->", color="#55607a", lw=1))

fig.text(0.01, 0.012, "GBC073 Inteligência Computacional (FACOM/UFU) | Caike Araujo, Matheus Izidro",
          fontsize=9, color="#55607a")

fig.savefig("slide.pdf")
print("slide.pdf gerado.")
print("Variancias reais (48 camadas):", [round(v, 2) for v in variancias])
