"""
Pipeline de Ingestao e Pre-processamento
Projeto: Sistema de Recomendacao de Culturas Agricolas
Disciplina: Inteligencia Computacional
Autores: Caike e Izidro

Este script cuida apenas da etapa de dados: carregar o CSV, separar
treino/teste, aplicar a expansao de caracteristicas phi(X) e normalizar.
A rede neural (MLP) entra em uma etapa seguinte.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


COLUNAS_ENTRADA = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
COLUNA_ALVO = "label"


def carregar_dados(caminho_csv):
    """
    Carrega o dataset CSV e separa em variaveis de entrada (X) e alvo (y).

    Espera as colunas: N, P, K, temperature, humidity, ph, rainfall, label
    """
    dados = pd.read_csv(caminho_csv)

    X = dados[COLUNAS_ENTRADA].copy()
    y = dados[COLUNA_ALVO].copy()

    return X, y


def dividir_treino_teste(X, y, proporcao_teste=0.2, semente=42):
    """
    Divide os dados em treino e teste (80/20 por padrao).

    Usa stratify=y para manter a proporcao das 22 classes de cultura
    parecida nos dois conjuntos, ja que o dataset tem varias classes
    e algumas com poucas amostras.
    """
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=proporcao_teste,
        random_state=semente,
        stratify=y,
    )
    return X_treino, X_teste, y_treino, y_teste


def aplicar_smote(X_treino, y_treino, fator_multiplicacao=4, k_neighbors=5, semente=42):
    """
    Aumenta o treino com exemplos sinteticos via SMOTE, so no treino
    (nunca no teste, senao a validacao fica contaminada com dados
    sinteticos "vazando" pra avaliacao).

    Como o SMOTE cria exemplos interpolando entre um ponto e seus k
    vizinhos mais proximos DA MESMA CLASSE, a distancia usada pra achar
    esses vizinhos precisa ser justa entre as features - por isso
    padronizamos antes de rodar o SMOTE (senao rainfall, que varia em
    dezenas/centenas, dominaria a distancia sobre ph, que varia entre 3 e 10)
    e desfazemos a padronizacao depois, devolvendo os dados na escala
    original para o resto do pipeline (phi, normalizacao final) seguir
    igual ao fluxo sem SMOTE.

    fator_multiplicacao=4 significa: cada classe passa a ter ~4x mais
    exemplos no treino (100 amostras/classe no treino original ~= 80
    depois do split -> ~320 depois do SMOTE).
    """
    scaler_smote = StandardScaler()
    X_treino_escalado = scaler_smote.fit_transform(X_treino)

    contagem_atual = y_treino.value_counts()
    estrategia_amostragem = {
        classe: int(round(contagem * fator_multiplicacao))
        for classe, contagem in contagem_atual.items()
    }

    smote = SMOTE(
        sampling_strategy=estrategia_amostragem,
        k_neighbors=k_neighbors,
        random_state=semente,
    )
    X_aumentado_escalado, y_aumentado = smote.fit_resample(X_treino_escalado, y_treino)

    X_aumentado_array = scaler_smote.inverse_transform(X_aumentado_escalado)
    X_treino_aumentado = pd.DataFrame(X_aumentado_array, columns=X_treino.columns)
    y_treino_aumentado = pd.Series(y_aumentado, name=y_treino.name)

    return X_treino_aumentado, y_treino_aumentado


def validar_smote(X_original, y_original, X_aumentado, y_aumentado):
    """
    Validacoes de sanidade sobre os dados sinteticos gerados pelo SMOTE:

    1. Todas as classes devem ter o mesmo numero de exemplos antes e
       depois (garante que o fator de multiplicacao foi aplicado
       igualmente, sem favorecer nenhuma cultura).
    2. Nenhum valor deve ficar fora dos limites fisicos minimos (N, P, K,
       umidade, chuva, ph nao podem ser negativos) - como o SMOTE so
       interpola entre pontos reais (combinacao convexa), isso deveria
       valer sempre, mas testamos explicitamente.
    3. Media e desvio padrao por classe antes/depois devem ficar
       proximos (o SMOTE nao deveria distorcer o "centro" da nuvem de
       pontos de cada cultura).
    """
    print("\n--- Validacao do SMOTE ---")

    contagem_antes = y_original.value_counts()
    contagem_depois = y_aumentado.value_counts()
    print(f"Classes antes: {len(contagem_antes)} | depois: {len(contagem_depois)}")
    print(f"Exemplos antes: {contagem_antes.sum()} | depois: {contagem_depois.sum()}")

    tamanhos_iguais_por_classe = contagem_depois.nunique() == 1
    print(f"Todas as classes com o mesmo tamanho depois do SMOTE? {tamanhos_iguais_por_classe}")

    colunas_nao_negativas = ["N", "P", "K", "humidity", "rainfall", "ph"]
    minimos = X_aumentado[colunas_nao_negativas].min()
    valores_negativos = (minimos < 0).any()
    print(f"Algum valor negativo em colunas fisicamente nao-negativas? {valores_negativos}")
    if valores_negativos:
        raise ValueError(f"SMOTE gerou valores negativos invalidos:\n{minimos}")

    medias_antes = X_original.assign(label=y_original).groupby("label").mean()
    medias_depois = X_aumentado.assign(label=y_aumentado).groupby("label").mean()
    diferenca_media_relativa = (
        (medias_depois - medias_antes).abs() / medias_antes.abs()
    ).mean().mean()
    print(f"Diferenca media relativa (media por classe, antes vs depois): {diferenca_media_relativa:.4f}")

    print("--- Fim da validacao ---\n")


def phi(X):
    """
    Mapeamento phi(X): expande o conjunto de caracteristicas original
    adicionando colunas derivadas com sentido agronomico, seguindo a
    ideia do Teorema de Cover de projetar os dados em uma dimensao maior
    para facilitar a separacao das classes.

    Importante: essa funcao NAO tem "fit" - e so uma transformacao
    matematica direta em cima dos valores de entrada. Por isso pode ser
    aplicada em treino e teste sem risco de vazamento de dados.

    Novas colunas criadas:
    - temp_umidade : Temperatura * Umidade (interacao climatica)
    - ph_quadrado  : pH ao quadrado (resposta nao linear da planta ao pH)
    - N_por_P      : razao entre Nitrogenio e Fosforo
    - chuva_log    : log(chuva + 1) (suaviza chuvas muito altas)
    - K_quadrado   : Potassio ao quadrado
    """
    X_expandido = X.copy()

    # Termo cruzado: interacao entre temperatura e umidade
    X_expandido["temp_umidade"] = X["temperature"] * X["humidity"]

    # Termo quadratico: resposta nao linear ao pH do solo
    X_expandido["ph_quadrado"] = X["ph"] ** 2

    # Razao entre nutrientes (+1 no denominador evita divisao por zero)
    X_expandido["N_por_P"] = X["N"] / (X["P"] + 1)

    # Log da chuva (+1 antes do log evita log(0), que daria -Inf)
    X_expandido["chuva_log"] = np.log(X["rainfall"] + 1)

    # Termo quadratico do potassio
    X_expandido["K_quadrado"] = X["K"] ** 2

    return X_expandido


def normalizar_dados(X_treino, X_teste):
    """
    Padroniza os dados (media 0, variancia 1).

    Regra rigida: o StandardScaler e ajustado (fit) SOMENTE no treino.
    O teste e apenas transformado com os parametros aprendidos no treino,
    para nao vazar informacao do teste para o pre-processamento.
    """
    scaler = StandardScaler()
    scaler.fit(X_treino)

    X_treino_array = scaler.transform(X_treino)
    X_teste_array = scaler.transform(X_teste)

    # Devolve como DataFrame para manter os nomes das colunas
    X_treino_normalizado = pd.DataFrame(
        X_treino_array, columns=X_treino.columns, index=X_treino.index
    )
    X_teste_normalizado = pd.DataFrame(
        X_teste_array, columns=X_teste.columns, index=X_teste.index
    )

    return X_treino_normalizado, X_teste_normalizado, scaler


def checar_nan_inf(X, nome_conjunto):
    """
    Verificacao de seguranca: garante que nao existem valores
    NaN ou infinitos depois das transformacoes.

    Se encontrar algum valor invalido, interrompe a execucao com
    ValueError em vez de so avisar, para nao deixar a rede neural
    receber dados corrompidos (mesmo rigor do harness do Desafio 1,
    que usa assert para o mesmo fim).
    """
    tem_nan = X.isna().any().any()
    tem_inf = np.isinf(X.to_numpy()).any()

    if tem_nan or tem_inf:
        raise ValueError(f"Valores invalidos (NaN/Inf) encontrados em {nome_conjunto}!")

    print(f"{nome_conjunto}: nenhum valor NaN ou Inf encontrado.")


def executar_pipeline(caminho_csv, usar_smote=True, fator_multiplicacao_smote=4):
    """
    Executa o pipeline completo:
    carregar -> dividir -> [SMOTE no treino] -> aplicar phi -> normalizar -> checar dados.
    """
    print("1. Carregando dados...")
    X, y = carregar_dados(caminho_csv)
    print(f"   Total de amostras: {X.shape[0]}, colunas originais: {X.shape[1]}")

    print("2. Dividindo em treino (80%) e teste (20%)...")
    X_treino, X_teste, y_treino, y_teste = dividir_treino_teste(X, y)
    print(f"   Treino: {X_treino.shape[0]} amostras | Teste: {X_teste.shape[0]} amostras")

    if usar_smote:
        print(f"2.1. Aplicando SMOTE no treino (fator {fator_multiplicacao_smote}x)...")
        X_treino_original, y_treino_original = X_treino, y_treino
        X_treino, y_treino = aplicar_smote(
            X_treino, y_treino, fator_multiplicacao=fator_multiplicacao_smote
        )
        print(f"   Treino depois do SMOTE: {X_treino.shape[0]} amostras")
        validar_smote(X_treino_original, y_treino_original, X_treino, y_treino)

    print("3. Aplicando phi(X) - expansao de caracteristicas...")
    X_treino_phi = phi(X_treino)
    X_teste_phi = phi(X_teste)
    print(f"   Colunas depois de phi(X): {X_treino_phi.shape[1]}")

    print("4. Normalizando (fit apenas no treino)...")
    X_treino_final, X_teste_final, scaler = normalizar_dados(X_treino_phi, X_teste_phi)

    print("5. Checando valores invalidos...")
    checar_nan_inf(X_treino_final, "treino")
    checar_nan_inf(X_teste_final, "teste")

    print("\nPipeline concluido.")
    print(f"Formato final treino: {X_treino_final.shape}")
    print(f"Formato final teste : {X_teste_final.shape}")

    return X_treino_final, X_teste_final, y_treino, y_teste, scaler


if __name__ == "__main__":
    CAMINHO_DATASET = "dados/Crop_recommendation.csv"

    X_treino, X_teste, y_treino, y_teste, scaler = executar_pipeline(CAMINHO_DATASET)

    print("\nPrimeiras linhas do treino ja processado:")
    print(X_treino.head())