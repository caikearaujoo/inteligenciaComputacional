"""
Pipeline de Ingestao e Pre-processamento
Projeto: Classificacao de Cobertura Vegetal a partir de Solo e Relevo
Disciplina: Inteligencia Computacional
Autores: Caike e Izidro

Dataset: Forest Covertype (UCI). Cada linha e uma celula de 30 x 30 m da
Floresta Nacional Roosevelt, no norte do Colorado (EUA): 581.012 celulas,
54 variaveis e 7 classes de cobertura vegetal.

As 54 variaveis sao:
  - 10 numericas de relevo e distancias (elevacao, aspecto, declividade,
    distancias a agua, estrada e focos de incendio, 3 indices de sombra);
  - 4 binarias de area selvagem (one-hot);
  - 40 binarias de tipo de solo (one-hot).

TAREFA: dadas as condicoes de uma celula de terreno (solo e relevo), prever
qual e a cobertura vegetal dominante ali (aptidao do terreno).

Carregamento: `fetch_covtype` do scikit-learn, sem precisar baixar CSV a mao.

DESBALANCEAMENTO: Lodgepole Pine tem 283.301 celulas e Cottonwood/Willow so
2.747. Chutar sempre a classe majoritaria ja acerta 48,8%, entao a acuracia
simples engana - o projeto reporta tambem acuracia balanceada, F1 macro e
recall por classe.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


NOMES_CLASSES = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]

COLUNAS_NUMERICAS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
COLUNAS_AREA = [f"Wilderness_Area_{i}" for i in range(4)]
COLUNAS_SOLO = [f"Soil_Type_{i}" for i in range(40)]
COLUNAS_BINARIAS = COLUNAS_AREA + COLUNAS_SOLO


def carregar_dados():
    """
    Carrega o Covertype e devolve X (DataFrame, float32) e y (inteiros 0 a 6).

    O dataset original numera as classes de 1 a 7; subtraimos 1 porque o
    PyTorch espera rotulos comecando em 0.
    """
    dados = fetch_covtype(as_frame=True)
    X = dados.data.astype("float32")
    y = (dados.target.astype("int64") - 1).reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y


def dividir_treino_validacao_teste(X, y, prop_validacao=0.15, prop_teste=0.15, semente=42):
    """
    Divide em treino (70%), validacao (15%) e teste (15%), estratificado.

    Por que TRES partes e nao duas: ao longo do projeto vamos comparar
    ativacoes, inicializacoes, otimizadores e agendas de taxa de aprendizado. Toda decisao
    desse tipo e tomada olhando a VALIDACAO. O teste fica intocado ate o
    final, para ser uma medida honesta - se escolhessemos hiperparametros
    olhando o teste, ele deixaria de ser dado "nunca visto" e o resultado
    final ficaria otimista.

    A estratificacao mantem a proporcao das 7 classes nas tres partes, o que
    importa muito aqui: Cottonwood/Willow tem so 2.747 celulas no total.
    """
    X_resto, X_teste, y_resto, y_teste = train_test_split(
        X, y, test_size=prop_teste, random_state=semente, stratify=y
    )
    prop_val_do_resto = prop_validacao / (1.0 - prop_teste)
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X_resto, y_resto, test_size=prop_val_do_resto, random_state=semente, stratify=y_resto
    )
    return X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste


# Termos do phi que podem ser ligados/desligados nos experimentos (ablacao):
#   "aspecto"   -> aspecto vira seno e cosseno
#   "agua"      -> distancia euclidiana ate a agua
#   "elev_decl" -> elevacao x declividade
TERMOS_PHI = ("aspecto", "agua", "elev_decl")


def phi(X, termos=TERMOS_PHI):
    """
    Mapeamento phi(X): transforma o terreno bruto nas caracteristicas que a
    rede recebe, com sentido fisico (Teorema de Cover: projetar em outra
    representacao para facilitar a separacao das classes).

    Nao tem "fit" - e transformacao matematica direta, entao pode ser aplicada
    em treino, validacao e teste sem risco de vazamento.

    1. ASPECTO vira seno e cosseno. Aspecto e um angulo (direcao para onde a
       encosta aponta, de 0 a 359 graus). Para um numero comum, 1 grau e 359
       graus parecem muito distantes, mas na realidade sao vizinhos: as duas
       encostas apontam quase para o mesmo lado (norte). Seno e cosseno
       colocam o angulo num circulo, onde a vizinhanca fica correta.

    2. DISTANCIA EUCLIDIANA ATE A AGUA. O dado traz a distancia horizontal e a
       vertical ate a agua mais proxima, separadas. O que importa para a
       vegetacao e a distancia real, que e a hipotenusa dos dois.

    3. ELEVACAO x DECLIVIDADE. Interacao entre duas variaveis fortes: o mesmo
       declive tem efeito diferente em altitudes diferentes (neve, vento,
       drenagem), e um termo cruzado deixa a rede enxergar isso direto.
    """
    saida = pd.DataFrame(index=X.index)

    for coluna in COLUNAS_NUMERICAS:
        # com o termo "aspecto" ligado, o Aspect bruto e trocado por seno/cosseno
        if coluna != "Aspect" or "aspecto" not in termos:
            saida[coluna] = X[coluna]

    if "aspecto" in termos:
        angulo = np.deg2rad(X["Aspect"])
        saida["Aspect_sen"] = np.sin(angulo)
        saida["Aspect_cos"] = np.cos(angulo)

    if "agua" in termos:
        saida["Dist_Agua"] = np.sqrt(
            X["Horizontal_Distance_To_Hydrology"] ** 2
            + X["Vertical_Distance_To_Hydrology"] ** 2
        )

    if "elev_decl" in termos:
        saida["Elevacao_x_Declividade"] = X["Elevation"] * X["Slope"]

    for coluna in COLUNAS_BINARIAS:
        saida[coluna] = X[coluna]

    return saida.astype("float32")


def colunas_continuas(X):
    """Colunas que precisam ser padronizadas: todas menos as 44 binarias."""
    return [c for c in X.columns if c not in COLUNAS_BINARIAS]


def normalizar_dados(X_treino, X_validacao, X_teste):
    """
    Padroniza so as colunas CONTINUAS (media 0, variancia 1), com fit apenas
    no treino. As 44 binarias ficam como estao: ja valem 0 ou 1, e padroniza-las
    so tiraria o significado de "este solo esta presente ou nao".
    """
    continuas = colunas_continuas(X_treino)
    scaler = StandardScaler()
    scaler.fit(X_treino[continuas])

    saidas = []
    for X in (X_treino, X_validacao, X_teste):
        X = X.copy()
        X[continuas] = scaler.transform(X[continuas])
        saidas.append(X.astype("float32"))

    return saidas[0], saidas[1], saidas[2], scaler


def checar_nan_inf(X, nome_conjunto):
    """
    Interrompe a execucao se sobrou NaN ou infinito depois das transformacoes.
    """
    if X.isna().any().any() or np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(f"Valores invalidos (NaN/Inf) encontrados em {nome_conjunto}!")
    print(f"{nome_conjunto}: nenhum valor NaN ou Inf encontrado.")


def executar_pipeline(termos_phi=TERMOS_PHI):
    """
    Pipeline completo:
    carregar -> dividir (70/15/15) -> phi -> normalizar -> checar.
    """
    print("1. Carregando o Covertype (fetch_covtype)...")
    X, y = carregar_dados()
    print(f"   Celulas: {X.shape[0]} | variaveis: {X.shape[1]} | classes: {y.nunique()}")
    majoritaria = y.value_counts(normalize=True).max()
    print(f"   Chutar sempre a classe majoritaria acerta {majoritaria * 100:.1f}%")

    print("2. Dividindo em treino (70%), validacao (15%) e teste (15%)...")
    X_tr, X_va, X_te, y_tr, y_va, y_te = dividir_treino_validacao_teste(X, y)
    print(f"   Treino: {len(X_tr)} | Validacao: {len(X_va)} | Teste: {len(X_te)}")

    print("3. Aplicando phi(X)...")
    X_tr, X_va, X_te = phi(X_tr, termos_phi), phi(X_va, termos_phi), phi(X_te, termos_phi)
    print(f"   Colunas: {X.shape[1]} -> {X_tr.shape[1]} "
          f"(aspecto vira seno/cosseno, +distancia a agua, +elevacao x declividade)")

    print("4. Normalizando so as colunas continuas (fit apenas no treino)...")
    X_tr, X_va, X_te, scaler = normalizar_dados(X_tr, X_va, X_te)
    print(f"   Continuas padronizadas: {len(colunas_continuas(X_tr))} | "
          f"binarias mantidas: {len(COLUNAS_BINARIAS)}")

    print("5. Checando valores invalidos...")
    checar_nan_inf(X_tr, "treino")
    checar_nan_inf(X_va, "validacao")
    checar_nan_inf(X_te, "teste")

    print("\nPipeline concluido.")
    print(f"Formato final treino    : {X_tr.shape}")
    print(f"Formato final validacao : {X_va.shape}")
    print(f"Formato final teste     : {X_te.shape}")

    return X_tr, X_va, X_te, y_tr.to_numpy(), y_va.to_numpy(), y_te.to_numpy(), scaler


if __name__ == "__main__":
    X_tr, X_va, X_te, y_tr, y_va, y_te, scaler = executar_pipeline()

    print("\nPrimeiras linhas do treino ja processado (colunas continuas):")
    print(X_tr[colunas_continuas(X_tr)].head())
