"""
Pipeline de Ingestao e Pre-processamento
Projeto: Sistema de Recomendacao de Culturas Agricolas
Disciplina: Inteligencia Computacional
Autores: Caike e Izidro

Dataset: "Crop Recommendation using Soil Properties and Weather Prediction"
(Mendeley Data, DOI 10.17632/8v757rr4st.1) - dados REAIS de campo da Agencia
de Transformacao Agricola da Etiopia (solo e cultura) combinados com dados
climaticos da NASA. 3.867 registros, 12 culturas, 28 variaveis.

Substituiu o dataset sintetico do Kaggle (2.200 linhas) usado ate a Semana 2 -
ver TrabalhoFinal.md para a justificativa completa da troca.

Este script cuida apenas da etapa de dados: carregar, limpar, separar
treino/teste, balancear, expandir caracteristicas e normalizar.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


COLUNA_COR = "Soilcolor"
COLUNAS_SOLO = ["Ph", "K", "P", "N", "Zn", "S"]
COLUNAS_CLIMA = [
    "QV2M-W", "QV2M-Sp", "QV2M-Su", "QV2M-Au",
    "T2M_MAX-W", "T2M_MAX-Sp", "T2M_MAX-Su", "T2M_MAX-Au",
    "T2M_MIN-W", "T2M_MIN-Sp", "T2M_MIN-Su", "T2M_MIN-Au",
    "PRECTOTCORR-W", "PRECTOTCORR-Sp", "PRECTOTCORR-Su", "PRECTOTCORR-Au",
    "WD10M", "GWETTOP", "CLOUD_AMT", "WS2M_RANGE", "PS",
]
COLUNA_ALVO = "label"

# Categorias canonicas da cor do solo, fixadas para que treino e teste gerem
# exatamente as mesmas colunas no one-hot.
CORES_CANONICAS = ["black", "brown", "gray", "red", "outro"]

# Colunas fortemente assimetricas (cauda longa a direita), medido no dataset:
# P tem mediana 4 e maximo 782; Zn mediana 1,5 e maximo 45,5. O log comprime
# essa cauda e evita que poucos valores extremos dominem a normalizacao.
COLUNAS_ASSIMETRICAS = ["K", "P", "Zn", "S"]


def carregar_dados(caminho_csv):
    """
    Carrega o CSV e separa entrada (X) e alvo (y).

    X mantem a cor do solo como texto nesta etapa; a limpeza e a codificacao
    acontecem depois, em passos proprios.
    """
    dados = pd.read_csv(caminho_csv)

    colunas_entrada = [COLUNA_COR] + COLUNAS_SOLO + COLUNAS_CLIMA
    X = dados[colunas_entrada].copy()
    y = dados[COLUNA_ALVO].copy()

    return X, y


def limpar_cor_solo(coluna_cor):
    """
    Normaliza a cor do solo, que no dado bruto tem 45 variantes escritas a mao
    para o que sao ~5 cores: "reddish brown", "Reddish brown", "Redish brown",
    "Reddis brown", "Reddish broown", "Redishbrown" sao todas a mesma coisa.

    Regra usada para escolher a cor: vale a ULTIMA cor citada no texto, porque
    em ingles o substantivo vem depois do adjetivo - "reddish brown" e um
    MARROM avermelhado (marrom), e "reddish gray" e um CINZA avermelhado
    (cinza). Pegar a primeira cor daria a resposta errada nos dois casos.

    Texto sem nenhuma cor reconhecivel (ex.: "other", "replacement of
    inaccessible target") vai para a categoria "outro".
    """
    trocas = {
        "redish": "reddish",
        "reddis ": "reddish ",
        "broown": "brown",
        "lihgtish": "lightish",
        "greyish": "grayish",
        "grey": "gray",
        "darkbrown": "dark brown",
        "redishbrown": "reddish brown",
    }
    cores_buscadas = ["black", "brown", "gray", "red", "yellow"]

    def classificar(valor):
        texto = str(valor).lower().strip()
        for errado, certo in trocas.items():
            texto = texto.replace(errado, certo)

        posicoes = [(texto.rfind(cor), cor) for cor in cores_buscadas if cor in texto]
        if not posicoes:
            return "outro"

        cor = max(posicoes)[1]
        # "yellow" so aparece em "yellowish brown", que ja vira brown pela
        # regra da ultima cor; se aparecesse sozinho, cai em "outro" por nao
        # ter volume suficiente para virar categoria propria.
        return cor if cor in CORES_CANONICAS else "outro"

    return coluna_cor.map(classificar)


def dividir_treino_teste(X, y, proporcao_teste=0.2, semente=42):
    """
    Divide em treino e teste (80/20), estratificado pelas 12 culturas.

    A estratificacao e essencial aqui porque o dataset e DESBALANCEADO de
    verdade: Teff tem 1.260 exemplos e Fallow tem 26. Sem estratificar, as
    classes raras poderiam ficar inteiras de um lado so.
    """
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=proporcao_teste,
        random_state=semente,
        stratify=y,
    )
    return X_treino, X_teste, y_treino, y_teste


def aplicar_smote(X_treino, y_treino, fator_maximo=5, k_neighbors=5, semente=42):
    """
    Balanceia o treino com exemplos sinteticos, SO no treino.

    Duas decisoes importantes aqui:

    1. Usamos SMOTENC, nao o SMOTE comum. O SMOTE comum interpola entre
       pontos, o que funciona para numeros mas nao para categorias: a cor do
       solo viraria "0,6 de marrom", que nao existe. O SMOTENC trata as
       colunas categoricas separadamente, escolhendo a categoria mais comum
       entre os vizinhos em vez de interpolar.

    2. Limitamos o aumento a `fator_maximo` vezes o tamanho original de cada
       classe, em vez de igualar tudo a classe majoritaria. Motivo: Fallow tem
       ~21 exemplos no treino e Teff tem ~1.008. Igualar tudo exigiria gerar
       987 exemplos sinteticos a partir de 21 reais (47x) - a rede aprenderia
       basicamente ruido interpolado. Com o limite de 5x, o desbalanceamento
       diminui sem inventar dado demais.
    """
    contagem = y_treino.value_counts()
    maior = contagem.max()
    estrategia = {
        classe: int(min(maior, n * fator_maximo))
        for classe, n in contagem.items()
    }

    # o SMOTENC precisa saber QUAIS colunas sao categoricas, por posicao
    indice_categorico = [X_treino.columns.get_loc(COLUNA_COR)]

    smote = SMOTENC(
        categorical_features=indice_categorico,
        sampling_strategy=estrategia,
        k_neighbors=k_neighbors,
        random_state=semente,
    )
    X_aumentado, y_aumentado = smote.fit_resample(X_treino, y_treino)

    return (
        pd.DataFrame(X_aumentado, columns=X_treino.columns),
        pd.Series(y_aumentado, name=y_treino.name),
    )


def validar_smote(y_antes, y_depois):
    """
    Confere que o balanceamento fez o que deveria: nenhuma classe encolheu, e
    a razao entre a maior e a menor classe diminuiu.
    """
    print("\n--- Validacao do balanceamento ---")
    antes, depois = y_antes.value_counts(), y_depois.value_counts()

    razao_antes = antes.max() / antes.min()
    razao_depois = depois.max() / depois.min()
    print(f"Exemplos: {antes.sum()} -> {depois.sum()}")
    print(f"Menor classe: {antes.min()} -> {depois.min()}")
    print(f"Desbalanceamento (maior/menor): {razao_antes:.1f}x -> {razao_depois:.1f}x")

    # alinha pelo nome da classe: as duas contagens vem ordenadas por
    # frequencia, e essa ordem muda depois do balanceamento
    encolheu = (depois.reindex(antes.index) < antes).any()
    if encolheu:
        raise ValueError("Alguma classe perdeu exemplos no balanceamento.")
    print("Nenhuma classe perdeu exemplos.")
    print("--- Fim da validacao ---\n")


def phi(X):
    """
    Mapeamento phi(X): expande as caracteristicas com colunas derivadas,
    seguindo a ideia do Teorema de Cover (projetar em dimensao maior para
    facilitar a separacao das classes).

    Nao tem "fit" - e transformacao matematica direta, entao pode ser aplicada
    em treino e teste sem risco de vazamento.

    As transformacoes aqui sao justificadas pelos dados, nao escolhidas no
    chute (ver estatisticas no TrabalhoFinal.md):

    - log dos nutrientes assimetricos (K, P, Zn, S): P vai de 0 a 782 com
      mediana 4, ou seja, alguns poucos pontos extremos dominariam a escala.
    - N_por_P: balanco entre nitrogenio e fosforo (razao entre nutrientes).
    - chuva_total: soma das 4 estacoes, o volume anual de chuva.
    - amplitude_termica: media das maximas menos media das minimas, o quanto
      a temperatura varia ao longo do ano.
    """
    X_expandido = X.copy()

    for coluna in COLUNAS_ASSIMETRICAS:
        X_expandido[f"log_{coluna}"] = np.log1p(X[coluna])

    X_expandido["N_por_P"] = X["N"] / (X["P"] + 1)

    colunas_chuva = [c for c in COLUNAS_CLIMA if c.startswith("PRECTOTCORR")]
    X_expandido["chuva_total"] = X[colunas_chuva].sum(axis=1)

    maximas = [c for c in COLUNAS_CLIMA if c.startswith("T2M_MAX")]
    minimas = [c for c in COLUNAS_CLIMA if c.startswith("T2M_MIN")]
    X_expandido["amplitude_termica"] = X[maximas].mean(axis=1) - X[minimas].mean(axis=1)

    return X_expandido


def codificar_cor(X):
    """
    Transforma a cor do solo (categoria) em colunas 0/1 (one-hot).

    Usa categorias FIXAS (CORES_CANONICAS) para garantir que treino e teste
    produzam exatamente as mesmas colunas, na mesma ordem - se uma cor rara
    nao aparecesse no teste, sem isso o teste teria uma coluna a menos e a
    rede quebraria.
    """
    X = X.copy()
    X[COLUNA_COR] = pd.Categorical(X[COLUNA_COR], categories=CORES_CANONICAS)
    return pd.get_dummies(X, columns=[COLUNA_COR], prefix="cor", dtype=float)


def normalizar_dados(X_treino, X_teste):
    """
    Padroniza os dados (media 0, variancia 1).

    O StandardScaler e ajustado (fit) SOMENTE no treino; o teste e apenas
    transformado. Ajustar no teste vazaria informacao dele para a preparacao.
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
    Converte o rotulo de texto ("Teff", "Maize", ...) para inteiro (0 a 11),
    que e o formato esperado pela rede com Softmax/CrossEntropy.
    """
    encoder = LabelEncoder()
    y_treino_cod = encoder.fit_transform(y_treino)
    y_teste_cod = encoder.transform(y_teste)
    return y_treino_cod, y_teste_cod, encoder


def checar_nan_inf(X, nome_conjunto):
    """
    Interrompe a execucao se sobrou algum NaN ou infinito depois das
    transformacoes, para a rede nunca receber dado corrompido.
    """
    tem_nan = X.isna().any().any()
    tem_inf = np.isinf(X.to_numpy(dtype=float)).any()

    if tem_nan or tem_inf:
        raise ValueError(f"Valores invalidos (NaN/Inf) encontrados em {nome_conjunto}!")

    print(f"{nome_conjunto}: nenhum valor NaN ou Inf encontrado.")


def executar_pipeline(caminho_csv, usar_smote=True, fator_maximo_smote=5):
    """
    Pipeline completo:
    carregar -> limpar cor -> dividir -> [balancear] -> phi -> one-hot ->
    normalizar -> checar -> codificar rotulo.
    """
    print("1. Carregando dados...")
    X, y = carregar_dados(caminho_csv)
    print(f"   Amostras: {X.shape[0]} | colunas: {X.shape[1]} | culturas: {y.nunique()}")

    print("2. Limpando a cor do solo...")
    variantes_antes = X[COLUNA_COR].nunique()
    X[COLUNA_COR] = limpar_cor_solo(X[COLUNA_COR])
    print(f"   Variantes de texto: {variantes_antes} -> {X[COLUNA_COR].nunique()} categorias")

    print("3. Dividindo em treino (80%) e teste (20%)...")
    X_treino, X_teste, y_treino, y_teste = dividir_treino_teste(X, y)
    print(f"   Treino: {X_treino.shape[0]} | Teste: {X_teste.shape[0]}")

    if usar_smote:
        print(f"4. Balanceando o treino (SMOTENC, limite de {fator_maximo_smote}x)...")
        y_treino_antes = y_treino
        X_treino, y_treino = aplicar_smote(
            X_treino, y_treino, fator_maximo=fator_maximo_smote
        )
        validar_smote(y_treino_antes, y_treino)

    print("5. Aplicando phi(X) - expansao de caracteristicas...")
    X_treino = phi(X_treino)
    X_teste = phi(X_teste)

    print("6. Codificando a cor do solo (one-hot)...")
    X_treino = codificar_cor(X_treino)
    X_teste = codificar_cor(X_teste)
    print(f"   Colunas finais: {X_treino.shape[1]}")

    print("7. Normalizando (fit apenas no treino)...")
    X_treino_final, X_teste_final, scaler = normalizar_dados(X_treino, X_teste)

    print("8. Checando valores invalidos...")
    checar_nan_inf(X_treino_final, "treino")
    checar_nan_inf(X_teste_final, "teste")

    print("9. Codificando rotulo (texto -> inteiro)...")
    y_treino_cod, y_teste_cod, encoder_rotulo = codificar_rotulo(y_treino, y_teste)
    print(f"   Classes: {len(encoder_rotulo.classes_)}")

    print("\nPipeline concluido.")
    print(f"Formato final treino: {X_treino_final.shape}")
    print(f"Formato final teste : {X_teste_final.shape}")

    return X_treino_final, X_teste_final, y_treino_cod, y_teste_cod, scaler, encoder_rotulo


if __name__ == "__main__":
    CAMINHO_DATASET = "dados/Crop_Ethiopia_soil_weather.csv"

    X_treino, X_teste, y_treino, y_teste, scaler, encoder_rotulo = executar_pipeline(
        CAMINHO_DATASET
    )

    print("\nPrimeiras linhas do treino ja processado:")
    print(X_treino.head())
