"""
Pipeline de Ingestao e Pre-processamento
Projeto: Previsao de Produtividade Agricola (Inteligencia Computacional)
Autores: Caike e Izidro

Dataset: dados agricolas oficiais da India (19.689 registros, 1997-2020),
por estado, ano, safra e cultura. Producao e area vem do orgao de estatistica
agricola; chuva, fertilizante e pesticida acompanham cada registro.

TAREFA: dadas as condicoes de uma lavoura (estado, safra, chuva, insumos,
area) e a cultura considerada, prever em que FAIXA de produtividade ela deve
ficar - baixa, media ou alta - em relacao ao tipico daquela cultura.

Por que faixa relativa, e nao o valor absoluto: a coluna Yield esta em
unidades DIFERENTES conforme a cultura (coco e medido em numero de cocos por
hectare, cana em toneladas). Comparar 6.317 de coco com 3,4 de arroz nao
significa nada. Dividindo cada valor pela mediana historica da propria
cultura, a unidade se cancela e a comparacao passa a ser justa: "essa lavoura
rende mais ou menos que o tipico dessa cultura?".
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


COLUNA_CULTURA = "Crop"
COLUNA_ALVO_BRUTO = "Yield"
COLUNAS_CATEGORICAS = ["Crop", "Season", "State"]
COLUNAS_NUMERICAS = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]

# Production NAO entra como entrada: Yield = Production / Area, entao usa-la
# seria entregar a resposta ao modelo (conferido no dado: correlacao 0,9965
# entre Yield e Production/Area).
COLUNA_VAZAMENTO = "Production"

# Fronteiras das faixas, em fracao da mediana historica da cultura.
# Abaixo de 0,75 = baixa; entre 0,75 e 1,25 = media; acima = alta.
LIMITE_BAIXA = 0.75
LIMITE_ALTA = 1.25
NOMES_FAIXAS = ["baixa", "media", "alta"]

# Culturas com menos exemplos que isso sao agrupadas em "Outras": com poucos
# registros nao da para estimar uma mediana historica confiavel.
MINIMO_POR_CULTURA = 100
ROTULO_OUTRAS = "Outras"


def carregar_dados(caminho_csv):
    """
    Carrega o CSV e devolve as colunas de entrada mais a produtividade bruta.

    A produtividade ainda nao e o rotulo final: ela vira faixa (baixa/media/
    alta) depois do split, em `criar_faixas_produtividade`.
    """
    dados = pd.read_csv(caminho_csv)
    dados = dados.drop(columns=[COLUNA_VAZAMENTO])

    # espacos sobrando no nome da safra ("Kharif     ") atrapalhariam o one-hot
    dados["Season"] = dados["Season"].str.strip()

    return dados


def agrupar_culturas_raras(dados, minimo=MINIMO_POR_CULTURA):
    """
    Junta as culturas com poucos registros numa categoria "Outras".

    O rotulo deste projeto e definido POR CULTURA (produtividade relativa a
    mediana historica dela). Com 10 ou 30 registros, essa mediana e instavel:
    um unico ano atipico desloca a referencia e reclassifica todas as linhas
    daquela cultura. Agrupar evita construir o alvo em cima de estatistica
    fragil.
    """
    contagem = dados[COLUNA_CULTURA].value_counts()
    raras = contagem[contagem < minimo].index

    if len(raras) > 0:
        dados = dados.copy()
        dados[COLUNA_CULTURA] = dados[COLUNA_CULTURA].where(
            ~dados[COLUNA_CULTURA].isin(raras), ROTULO_OUTRAS
        )

    return dados, list(raras)


def dividir_treino_teste(dados, proporcao_teste=0.2, semente=42):
    """
    Divide em treino e teste (80/20), estratificado pela CULTURA.

    Estratificar pela cultura (e nao pela faixa) garante que toda cultura
    apareca no treino - necessario porque a mediana de referencia de cada uma
    e calculada so com dados de treino. Sem isso, uma cultura que caisse
    inteira no teste ficaria sem referencia.
    """
    return train_test_split(
        dados,
        test_size=proporcao_teste,
        random_state=semente,
        stratify=dados[COLUNA_CULTURA],
    )


def criar_faixas_produtividade(dados_treino, dados_teste):
    """
    Transforma a produtividade bruta em faixa (baixa/media/alta), relativa a
    mediana historica de cada cultura.

    Ponto critico de metodologia: as medianas sao calculadas SO COM O TREINO.
    Calcula-las no dataset inteiro usaria informacao do teste para definir o
    proprio rotulo - uma forma sutil de vazamento, que inflaria o resultado
    sem que isso aparecesse em nenhuma metrica.
    """
    medianas = dados_treino.groupby(COLUNA_CULTURA)[COLUNA_ALVO_BRUTO].median()

    def faixas(dados):
        referencia = dados[COLUNA_CULTURA].map(medianas)
        relativo = dados[COLUNA_ALVO_BRUTO] / referencia
        return pd.cut(
            relativo,
            bins=[-np.inf, LIMITE_BAIXA, LIMITE_ALTA, np.inf],
            labels=NOMES_FAIXAS,
        )

    return faixas(dados_treino), faixas(dados_teste), medianas


def phi(dados):
    """
    Mapeamento phi(X): monta as caracteristicas que a rede recebe.

    Duas transformacoes com motivo agronomico, nao escolhidas no chute:

    1. INSUMO POR HECTARE. Fertilizante e pesticida vem como totais da
       lavoura, entao crescem junto com a area - uma fazenda grande usa mais
       adubo sem necessariamente adubar melhor. O que afeta a produtividade e
       a dose por hectare, entao dividimos pela area.

    2. LOG das variaveis muito assimetricas. Area vai de alguns hectares a 50
       milhoes (mediana 9.317), e fertilizante chega a 4,8 bilhoes com mediana
       1,2 milhao. Sem o log, um punhado de lavouras gigantes dominaria a
       normalizacao e comprimiria todo o resto perto de zero.
    """
    X = pd.DataFrame(index=dados.index)

    area = dados["Area"].clip(lower=1)   # evita divisao por zero

    X["log_area"] = np.log1p(area)
    X["chuva_anual"] = dados["Annual_Rainfall"]
    X["log_fertilizante_ha"] = np.log1p(dados["Fertilizer"] / area)
    X["log_pesticida_ha"] = np.log1p(dados["Pesticide"] / area)
    X["ano"] = dados["Crop_Year"]

    for coluna in COLUNAS_CATEGORICAS:
        X[coluna] = dados[coluna]

    return X


def codificar_categoricas(X_treino, X_teste):
    """
    Converte cultura, safra e estado em colunas 0/1 (one-hot).

    As categorias sao fixadas pelas que aparecem no TREINO, e o teste e
    reindexado nessas mesmas colunas. Assim os dois conjuntos sempre tem as
    mesmas colunas, na mesma ordem - se uma categoria so aparecesse no teste,
    a rede receberia uma entrada que nunca viu e quebraria.
    """
    treino = pd.get_dummies(X_treino, columns=COLUNAS_CATEGORICAS, dtype=float)
    teste = pd.get_dummies(X_teste, columns=COLUNAS_CATEGORICAS, dtype=float)
    teste = teste.reindex(columns=treino.columns, fill_value=0.0)
    return treino, teste


def normalizar_dados(X_treino, X_teste):
    """
    Padroniza os dados (media 0, variancia 1), com fit SOMENTE no treino.
    """
    scaler = StandardScaler()
    scaler.fit(X_treino)

    X_treino_normalizado = pd.DataFrame(
        scaler.transform(X_treino), columns=X_treino.columns, index=X_treino.index
    )
    X_teste_normalizado = pd.DataFrame(
        scaler.transform(X_teste), columns=X_teste.columns, index=X_teste.index
    )
    return X_treino_normalizado, X_teste_normalizado, scaler


def codificar_rotulo(y_treino, y_teste):
    """
    Converte a faixa ("baixa"/"media"/"alta") em inteiro (0, 1, 2).
    """
    encoder = LabelEncoder()
    y_treino_cod = encoder.fit_transform(y_treino)
    y_teste_cod = encoder.transform(y_teste)
    return y_treino_cod, y_teste_cod, encoder


def checar_nan_inf(X, nome_conjunto):
    """
    Interrompe a execucao se sobrou NaN ou infinito depois das transformacoes.
    """
    tem_nan = X.isna().any().any()
    tem_inf = np.isinf(X.to_numpy(dtype=float)).any()

    if tem_nan or tem_inf:
        raise ValueError(f"Valores invalidos (NaN/Inf) encontrados em {nome_conjunto}!")

    print(f"{nome_conjunto}: nenhum valor NaN ou Inf encontrado.")


def executar_pipeline(caminho_csv):
    """
    Pipeline completo:
    carregar -> agrupar culturas raras -> dividir -> criar faixas (medianas so
    do treino) -> phi -> one-hot -> normalizar -> checar -> codificar rotulo.
    """
    print("1. Carregando dados...")
    dados = carregar_dados(caminho_csv)
    print(f"   Registros: {dados.shape[0]} | culturas: {dados[COLUNA_CULTURA].nunique()} "
          f"| estados: {dados['State'].nunique()} | anos: {dados['Crop_Year'].min()}-{dados['Crop_Year'].max()}")

    print("2. Agrupando culturas raras...")
    dados, raras = agrupar_culturas_raras(dados)
    print(f"   Agrupadas ({len(raras)}) em '{ROTULO_OUTRAS}' | culturas restantes: "
          f"{dados[COLUNA_CULTURA].nunique()}")

    print("3. Dividindo em treino (80%) e teste (20%)...")
    dados_treino, dados_teste = dividir_treino_teste(dados)
    print(f"   Treino: {len(dados_treino)} | Teste: {len(dados_teste)}")

    print("4. Criando as faixas de produtividade (medianas so do treino)...")
    y_treino, y_teste, medianas = criar_faixas_produtividade(dados_treino, dados_teste)
    distribuicao = y_treino.value_counts().reindex(NOMES_FAIXAS)
    print(f"   Distribuicao no treino: {distribuicao.to_dict()}")

    print("5. Aplicando phi(X)...")
    X_treino = phi(dados_treino)
    X_teste = phi(dados_teste)

    print("6. Codificando categoricas (one-hot)...")
    X_treino, X_teste = codificar_categoricas(X_treino, X_teste)
    print(f"   Colunas finais: {X_treino.shape[1]}")

    print("7. Normalizando (fit apenas no treino)...")
    X_treino_final, X_teste_final, scaler = normalizar_dados(X_treino, X_teste)

    print("8. Checando valores invalidos...")
    checar_nan_inf(X_treino_final, "treino")
    checar_nan_inf(X_teste_final, "teste")

    print("9. Codificando rotulo (faixa -> inteiro)...")
    y_treino_cod, y_teste_cod, encoder_rotulo = codificar_rotulo(y_treino, y_teste)
    print(f"   Faixas: {list(encoder_rotulo.classes_)}")

    print("\nPipeline concluido.")
    print(f"Formato final treino: {X_treino_final.shape}")
    print(f"Formato final teste : {X_teste_final.shape}")

    return X_treino_final, X_teste_final, y_treino_cod, y_teste_cod, scaler, encoder_rotulo


if __name__ == "__main__":
    CAMINHO_DATASET = "dados/Crop_Yield_India.csv"

    X_treino, X_teste, y_treino, y_teste, scaler, encoder_rotulo = executar_pipeline(
        CAMINHO_DATASET
    )

    print("\nPrimeiras linhas do treino ja processado:")
    print(X_treino.head())
