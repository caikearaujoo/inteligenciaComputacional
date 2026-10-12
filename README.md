# Inteligência Computacional: Desafios e Projeto Aplicado

Este repositório reúne o desenvolvimento prático da disciplina de Inteligência
Computacional (GBC073, FACOM/UFU), em duas frentes: as soluções dos **desafios
semanais** propostos em aula e o **projeto final**.

Autores: Caike Cesar Mota de Araújo e Matheus Izidro Campos dos Santos.

---

## Projeto final: classificação de cobertura vegetal

Uma Rede Neural Artificial (Multi-Layer Perceptron, em PyTorch) que recebe as
condições de solo e relevo de uma célula de terreno de 30 x 30 m e classifica
qual das 7 coberturas vegetais domina ali.

**Dados.** Forest Covertype (UCI), carregado direto pelo `fetch_covtype` do
scikit-learn: 581.012 células da Floresta Nacional Roosevelt (Colorado, EUA),
54 variáveis (10 numéricas de relevo e distâncias, 4 de área selvagem e 40 de
tipo de solo) e 7 classes. Os dados são bem desbalanceados: a classe mais comum
tem 283.301 células e a menos comum, 2.747. Chutar sempre a classe mais comum
acerta 48,8%, então a acurácia sozinha engana e o projeto reporta também
acurácia balanceada, F1 macro e recall por classe.

**Pipeline** (`trabalhoFinal/preprocessamento.py`):
- divisão estratificada em treino (70%), validação (15%) e teste (15%): a
  validação decide as escolhas e o teste só é medido no final;
- mapeamento de características `phi(X)` com sentido físico: o aspecto (um
  ângulo) vira seno e cosseno, a distância horizontal e a vertical até a água
  viram a distância euclidiana, e há um termo cruzado elevação x declividade
  (54 para 57 colunas);
- padronização apenas das variáveis contínuas, ajustada somente no treino.

**Modelo** (`trabalhoFinal/modelo.py`): rede `57 -> 128 -> 64 -> 7`, com 16.135
parâmetros (cerca de 25 exemplos de treino por parâmetro), ativação tanh nas
camadas ocultas e logits na saída (o `CrossEntropyLoss` já aplica o Softmax por
dentro). Treinada com a **receita vencedora do Desafio 3** (Muon nas matrizes ocultas e AdamW nos
vieses e na saída, agenda warmup + cosseno, 40 épocas); a melhor época é escolhida pela
validação.

**Experimentos** (`trabalhoFinal/experimentos.py`): a ativação, a inicialização
e o `phi` foram escolhidos por medição, sempre na validação e com 3 sementes
por configuração. Esses experimentos foram feitos com Adam 1e-3, antes do Desafio 3. Entre as ativações testadas, tanh ficou à frente de GELU e
ReLU (diferenças de 3 desvios ou mais) e a SELU foi a pior. Para o `phi`, na
MLP nenhuma variante se distinguiu do "sem phi"; num modelo linear ele ajudou
um pouco, quase tudo pelo aspecto em seno/cosseno.

**Resultado (teste, medido uma única vez, com a configuração escolhida):**
94,4% de acurácia, 90,1% de acurácia balanceada e F1 macro de 0,910, contra
48,8% de quem chuta a classe majoritária e 91,3% / F1 0,869 da baseline
(Adam 1e-3, lr constante, mesmo orçamento). A classe mais difícil é Aspen
(recall de 0,80).

**Como rodar:**

```
cd trabalhoFinal
py -m pip install -r requirements.txt
py preprocessamento.py     # prepara os dados e mostra os formatos
py modelo.py               # treina (~8 min), avalia e salva matriz_confusao_teste.png
py -u experimentos.py phi ativacao confirmacao   # reproduz os experimentos de phi e ativação
```

O dataset é baixado automaticamente na primeira execução.

---

## Desafios semanais e como se conectam ao projeto

Cada desafio é validado por um *harness* (testador automatizado) do professor.
Abaixo, o que fizemos e o que de fato foi levado para o projeto.

**Desafio 1: mapa de características phi(X) e Teorema de Cover.** Implementamos
um mapa `phi` genérico (polinômios por eixo, produto das coordenadas, raio,
senos/cossenos e termos cruzados) que leva os dados de `d` para `d'` colunas,
com `d < d' <= 64`, para um Perceptron linear separar problemas que não eram
linearmente separáveis (`desafios/desafio01/`).
No projeto, o código desse desafio não é reaproveitado: o `phi(X)` do projeto é
uma função própria, escrita para este dataset. O que foi levado é a ideia e as
regras do desafio (o `phi` do projeto leva 54 para 57 colunas, é determinístico
e a execução é interrompida se aparecer NaN ou infinito). Medimos o efeito: na
MLP o `phi` quase não muda o resultado; o termo cruzado elevação x declividade,
o único no espírito de Cover, não teve efeito mensurável.

**Desafio 2: ativação e inicialização em redes profundas.** O problema é manter
a variância do sinal perto de 1 ao longo das camadas, para que não sumisse nem
explodisse em redes de até 48 camadas sem BatchNorm. Nossa solução usa ReLU e
calcula o tamanho inicial dos pesos por `σ² = 1 / (fan_in · E[f(z)²])`, com
`E[f(z)²]` estimado por amostragem Monte Carlo, e reduz pela metade a escala da
última camada (`desafios/desafio2/`). O método funciona para qualquer ativação.
No projeto, a inicialização calibrada por Monte Carlo é a mesma. Já a ativação
mudou: os experimentos mostraram que o tanh é melhor que o ReLU nesta rede, e o
fator de 0,5 na última camada foi retirado por não ter efeito medido. A
inicialização calibrada ganha da padrão do PyTorch no início do treino. A rede
do projeto tem só duas camadas ocultas, onde o problema de variância do desafio
não é crítico, então o efeito é pequeno.

**Desafio 3: otimizadores e agendas de taxa de aprendizado, observados por dentro.**
O problema é achar, no problema do projeto, a melhor combinação de otimizador e
agenda de lr e **justificá-la lendo os sinais do treino**: a perda, a norma do
gradiente, a norma dos parâmetros, a razão de atualização por camada e o lr
efetivo, registrados a cada 25 passos. Comparamos SGD (com e sem Nesterov), Adam,
AdamW e Muon (obrigatório) com 4 agendas (constante, cosseno, warmup + cosseno e
WSD), no protocolo do enunciado: lr ajustado por otimizador, orçamento fixo de
31.800 passos, confirmação com 3 sementes e teste usado uma única vez
(`desafio3/`). A vencedora é o **Muon (lr 0,03 nas matrizes ocultas, AdamW 0,01 no
resto) com agenda warmup + cosseno**: F1 macro de validação 0,9121 ± 0,0009
contra 0,8646 ± 0,0028 da baseline (Adam 1e-3, constante), e no teste 0,9104
contra 0,8689. O ganho de +0,0475 se decompõe em lr maior com agenda que decai
(+0,031), weight decay (+0,004) e Muon (+0,013). Com 200 épocas o ganho cai para
+0,019 (1 semente). Nos sinais, o passo do Muon é proporcional ao lr e à forma da
matriz (variação de 1 a 2%, contra 29 a 41% do SGD), e a execução instável (SGD
com lr 3,0) mostra "passo grande demais". O modelo final do projeto
(`trabalhoFinal/modelo.py`) usa essa receita. Cada linha de `desafio3/experimentos.csv`
traz, na coluna `comando`, o comando exato que a reproduz (rodar de dentro de `desafio3/`),
e os sinais de cada execução estão em `desafio3/logs/`.

---

## Estrutura do repositório

- `trabalhoFinal/`: pipeline de dados (`preprocessamento.py`), rede neural
  (`modelo.py`), experimentos (`experimentos.py`), dependências
  (`requirements.txt`), descrição do tema (`tema.txt`) e a matriz de confusão
  do último treino.
- `desafios/desafio01/`: submissão do Desafio 1.
- `desafios/desafio2/`: submissão do Desafio 2, o harness do professor e o slide.
- `desafio3/`: Desafio 3 (`treino.py` configurável, `experimentos.csv`, `logs/`,
  `painel.pdf` e a apresentação em `slide.pdf`, gerada por `slide_html.py`).
  O `treino.py` importa a rede e o pipeline de `trabalhoFinal/`, então as duas
  pastas precisam estar juntas.
