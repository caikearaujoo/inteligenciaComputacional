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
    Verificacao de seguranca: confirma que nao existem valores
    NaN ou infinitos depois das transformacoes.
    """
    tem_nan = X.isna().any().any()
    tem_inf = np.isinf(X.to_numpy()).any()

    if tem_nan or tem_inf:
        print(f"ATENCAO: valores invalidos encontrados em {nome_conjunto}!")
    else:
        print(f"{nome_conjunto}: nenhum valor NaN ou Inf encontrado.")


def executar_pipeline(caminho_csv):
    """
    Executa o pipeline completo:
    carregar -> dividir -> aplicar phi -> normalizar -> checar dados.
    """
    print("1. Carregando dados...")
    X, y = carregar_dados(caminho_csv)
    print(f"   Total de amostras: {X.shape[0]}, colunas originais: {X.shape[1]}")

    print("2. Dividindo em treino (80%) e teste (20%)...")
    X_treino, X_teste, y_treino, y_teste = dividir_treino_teste(X, y)
    print(f"   Treino: {X_treino.shape[0]} amostras | Teste: {X_teste.shape[0]} amostras")

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