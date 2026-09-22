# 🧠 Inteligência Computacional: Desafios e Projeto Aplicado

Este repositório documenta o desenvolvimento prático da disciplina de Inteligência Computacional. Ele está dividido em duas frentes de trabalho que se conectam: a resolução dos **desafios semanais** propostos em aula e o desenvolvimento do nosso **projeto final** voltado para *Crop Science* (Ciência Agrícola).

---

## 🌱 O Foco Principal: Preditor Neural para Recomendação Agrícola

O objetivo central deste repositório é abrigar o código-fonte do nosso projeto de conclusão da disciplina. Trata-se de um **Sistema de Recomendação de Culturas Agrícolas** baseado em Redes Neurais Artificiais (Multi-Layer Perceptron).

O projeto aplica Inteligência Computacional para resolver uma dor real do agronegócio e da agricultura de precisão: cruzar variáveis físico-químicas do solo (Nitrogênio, Fósforo, Potássio, pH) e fatores meteorológicos (Temperatura, Umidade, Precipitação) para recomendar a espécie botânica com maior probabilidade de adaptação e sucesso financeiro no plantio.

### Estratégia de Desenvolvimento do Projeto
Nosso MLP não é construído de forma aleatória. Toda a arquitetura da rede, o pré-processamento de dados e as funções de ativação são justificados com base na teoria aprendida e testada nos desafios semanais da disciplina.

---

## 🧪 Desafios Semanais

Toda semana, um novo desafio teórico-prático é lançado pelo professor e validado via *harness* (testador automatizado). Armazenamos aqui as nossas soluções. 

Mais do que apenas obter nota, **usamos os resultados desses desafios para tomar decisões de engenharia no projeto final**. 

Até o momento, integramos os seguintes aprendizados:

*   **Desafio 1 - Mapeamento $\phi$ e Teorema de Cover:** 
    *   *O que fizemos:* Implementamos expansões polinomiais e cruzamentos dimensionais para ajudar um Perceptron linear a resolver problemas não-linearmente separáveis.
    *   *Impacto no Projeto:* Criamos uma função de pré-processamento para o nosso projeto agrícola que gera interações agronômicas relevantes (como `Temperatura × Umidade` e `pH²`), facilitando o aprendizado da rede multiclasse.

*   **Desafio 2 - Estabilidade de Sinal (Inicialização de He e ReLU):** 
    *   *O que fizemos:* Investigamos o colapso da variância do sinal e a explosão/desaparecimento do gradiente em redes muito profundas ($L=48$) sem BatchNorm.
    *   *Impacto no Projeto:* Justificou a nossa escolha por uma arquitetura rasa e larga para os dados tabulares agrícolas, implementando estritamente a inicialização de pesos de *He* em conjunto com a ativação ReLU para garantir a estabilidade do treinamento.

---

## 📂 Estrutura do Repositório

*   `/projeto_final`: Contém o pipeline de dados, a arquitetura da rede e os notebooks de treinamento do recomendador agrícola.
*   `/desafios`: Contém as submissões semanais exigidas pela disciplina, separadas por tópicos (ex: Desafio 1 - Cover, Desafio 2 - Inicialização, etc).

---
*Desenvolvido como requisito prático para a disciplina de Inteligência Computacional.*