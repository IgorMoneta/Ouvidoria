import math
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


MANIFESTACOES = [
    ('M001', 'infraestrutura', 'calçada quebrada na rua central, dificultando a passagem de pedestres.'),
    ('M002', 'saúde', 'demora para conseguir consulta com clínico geral no posto do bairro.'),
    ('M003', 'infraestrutura', 'buraco enorme na Av. Brasil'),
    ('M004', 'segurança', 'falta de iluminação deixa a rua escura durante a noite.'),
    ('M005', 'educação', 'escola municipal está com salas sem ventiladores funcionando.'),
    ('M006', 'meio ambiente', 'lixo acumulado próximo ao córrego está atraindo insetos.'),
    ('M007', 'infraestrutura', 'semáforo do cruzamento principal apresenta defeito.'),
    ('M008', 'saúde', 'posto de saúde sem médico'),
    ('M009', 'segurança', 'moradores relatam aumento de furtos perto da praça.'),
    ('M010', 'educação', 'falta material escolar para as turmas do ensino fundamental.'),
    ('M011', 'meio ambiente', 'árvores precisam de poda na avenida do mercado.'),
    ('M012', 'infraestrutura', 'tampa de bueiro solta oferece risco para motociclistas.'),
    ('M013', 'saúde', 'medicamentos básicos estão em falta na unidade de saúde.'),
    ('M014', 'segurança', 'pedido de reforço de patrulhamento no bairro durante a madrugada.'),
    ('M015', 'educação', 'a biblioteca da escola precisa de computadores para os alunos.'),
    ('M016', 'meio ambiente', 'queimada em terreno baldio produz muita fumaça.'),
    ('M017', 'infraestrutura', 'asfalto todo esburacado da avenida principal'),
    ('M018', 'saúde', 'fila muito longa para atendimento odontológico no posto.'),
    ('M019', 'segurança', 'poste de iluminação pública apagado aumenta a sensação de insegurança.'),
    ('M020', 'educação', 'goteiras na sala de aula prejudicam as atividades em dias de chuva.'),
    ('M021', 'meio ambiente', 'descarte irregular de entulho em área verde do município.'),
    ('M022', 'saúde', 'falta atendimento no PSF'),
    ('M023', 'infraestrutura', 'chuva forte abriu uma cratera na avenida principal e os veículos precisam desviar.'),
    ('M024', 'segurança', 'iluminação pública insuficiente deixa a via perigosa à noite.'),
    ('M025', 'educação', 'a escola está sem professores substitutos para algumas disciplinas.'),
    ('M026', 'meio ambiente', 'acúmulo de resíduos em terreno próximo às casas precisa de limpeza.'),
    ('M027', 'infraestrutura', 'bueiro sem tampa na calçada pode causar acidentes.'),
    ('M028', 'saúde', 'unidade básica está sem remédios para pacientes que fazem tratamento contínuo.'),
    ('M029', 'segurança', 'moradores pedem presença policial mais frequente durante a noite.'),
    ('M030', 'educação', 'ventiladores quebrados deixam as salas de aula muito quentes.'),
    ('M031', 'saúde', 'lâmpada queimada na praça'),
    ('M032', 'meio ambiente', 'é necessária a poda de árvores que estão encostando nos fios.'),
    ('M033', 'infraestrutura', 'o semáforo do centro não está funcionando corretamente e causa congestionamento.'),
    ('M034', 'saúde', 'o atendimento médico do posto está demorando várias horas.'),
    ('M035', 'segurança', 'uma rua do bairro permanece completamente escura depois das 19 horas.'),
    ('M036', 'educação', 'faltam livros didáticos para parte dos estudantes da escola municipal. A situação tem causado transtornos frequentes à comunidade e os moradores solicitam providências, vistoria técnica, manutenção e acompanhamento do caso até que o problema seja resolvido de forma adequada. A situação tem causado transtornos frequentes à comunidade e os moradores solicitam providências, vistoria técnica, manutenção e acompanhamento do caso até que o problema seja resolvido de forma adequada. A situação tem causado transtornos frequentes à comunidade e os moradores solicitam providências, vistoria técnica, manutenção e acompanhamento do caso até que o problema seja resolvido de forma adequada.'),
    ('M037', 'meio ambiente', 'há descarte de lixo e entulho perto do rio, causando mau cheiro e atraindo animais. A situação tem causado transtornos frequentes à comunidade e os moradores solicitam providências, vistoria técnica, manutenção e acompanhamento do caso até que o problema seja resolvido de forma adequada. A situação tem causado transtornos frequentes à comunidade e os moradores solicitam providências, vistoria técnica, manutenção e acompanhamento do caso até que o problema seja resolvido de forma adequada. A situação tem causado transtornos frequentes à comunidade e os moradores solicitam providências, vistoria técnica, manutenção e acompanhamento do caso até que o problema seja resolvido de forma adequada.'),
    ('M038', 'infraestrutura', 'A avenida do bairro apresenta diversos problemas de pavimentação, com trechos desnivelados, buracos de tamanhos diferentes e pontos onde o asfalto já se desprendeu. Em dias de chuva, a água fica acumulada nessas áreas e dificulta a passagem de carros, motocicletas e ônibus. Moradores relatam que os veículos precisam reduzir muito a velocidade e fazer desvios, aumentando o risco de acidentes e causando transtornos para quem utiliza a via diariamente. Solicita-se avaliação técnica e reparo da pavimentação.'),
    ('M039', 'saúde', 'Os moradores desta região têm enfrentado dificuldades para conseguir atendimento na unidade básica de saúde. A equipe disponível não parece suficiente para a demanda atual, fazendo com que pacientes aguardem por muito tempo antes de serem atendidos. Pessoas idosas e cidadãos que precisam renovar receitas também relatam dificuldades para conseguir orientação. A comunidade solicita que a situação seja verificada e que sejam adotadas medidas para melhorar o fluxo de atendimento e a disponibilidade de profissionais.'),
    ('M040', 'meio ambiente', 'O terreno localizado próximo às residências vem recebendo descarte irregular de lixo, restos de construção e outros materiais. Com o passar dos dias, o volume de resíduos aumentou e passou a ocupar uma área considerável. Durante períodos de chuva, parte do material é levada para a rua e para a área próxima ao córrego. Moradores também relatam mau cheiro e presença de insetos e animais. Solicita-se fiscalização, limpeza do local e orientação aos responsáveis para evitar que novos descartes aconteçam.'),
]

MODELOS = {
    "Paraphrase Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
    "BGE Small PT": "BAAI/bge-small-pt-v1.5",
}

STOPWORDS = {
    "a","o","as","os","um","uma","de","do","da","dos","das","e","em",
    "no","na","nos","nas","para","por","com","que","ao","aos","à","às",
    "se","como","até"
}


def limpar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )
    texto = re.sub(r"[^a-z0-9\s-]", " ", texto)
    tokens = re.findall(r"\b[a-z0-9-]+\b", texto)
    return " ".join(
        token for token in tokens
        if token not in STOPWORDS
    )


@st.cache_resource
def carregar_modelo(nome_modelo):
    return SentenceTransformer(nome_modelo)


@st.cache_data
def gerar_embeddings(textos, nome_modelo):
    modelo = carregar_modelo(nome_modelo)
    return modelo.encode(
        list(textos),
        normalize_embeddings=True
    )


def detectar_duplicatas(textos, limiar=0.85, nome_modelo="paraphrase-multilingual-MiniLM-L12-v2"):
    embeddings = gerar_embeddings(tuple(textos), nome_modelo)
    matriz = cosine_similarity(embeddings)
    pares = []

    for i in range(len(textos)):
        for j in range(i + 1, len(textos)):
            if matriz[i, j] >= limiar:
                pares.append((i, j, float(matriz[i, j])))

    return pares, matriz


def cor_score(score):
    if score > 0.7:
        return "#2e7d32"
    if score > 0.5:
        return "#ef6c00"
    return "#616161"


def mostrar_resultado(row):
    cor = cor_score(row["Score"])
    st.markdown(
        f"""
        <div style="padding:12px;border-left:6px solid {cor};
        background-color:#f5f5f5;border-radius:6px;margin-bottom:10px;">
        <b>{row['ID']} — {row['Categoria']}</b><br>
        <b>Score: {row['Score']:.4f}</b><br>
        {row['Texto']}
        </div>
        """,
        unsafe_allow_html=True
    )


st.set_page_config(
    page_title="Ouvidoria Semântica",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Ouvidoria Semântica")
st.write(
    "Busca semântica, detecção de duplicatas e análise de manifestações cidadãs."
)

df = pd.DataFrame(
    MANIFESTACOES,
    columns=["ID", "Categoria", "Texto"]
)

st.sidebar.header("Configurações")

nome_modelo = st.sidebar.selectbox(
    "Modelo de embedding",
    list(MODELOS.keys())
)

top_k = st.sidebar.slider(
    "Top-k resultados",
    min_value=1,
    max_value=20,
    value=5
)

nome_modelo_real = MODELOS[nome_modelo]

abas = st.tabs([
    "Busca Semântica",
    "Base Completa",
    "Espaço Vetorial",
    "Chunking"
])

with abas[0]:
    st.subheader("Busca Semântica")

    consulta = st.text_area(
        "Descrição da manifestação",
        placeholder="Digite uma descrição livre do problema..."
    )

    if consulta.strip():
        textos = df["Texto"].tolist()
        embeddings = gerar_embeddings(
            tuple(textos),
            nome_modelo_real
        )
        modelo = carregar_modelo(nome_modelo_real)
        vetor_consulta = modelo.encode(
            [consulta],
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            vetor_consulta,
            embeddings
        )[0]

        resultados = df.copy()
        resultados["Score"] = scores
        resultados = resultados.sort_values(
            "Score",
            ascending=False
        ).head(top_k)

        for _, row in resultados.iterrows():
            mostrar_resultado(row)

with abas[1]:
    st.subheader("Base Completa")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Gerar matriz de similaridade"):
        _, matriz = detectar_duplicatas(
            df["Texto"].tolist(),
            limiar=1.1,
            nome_modelo=nome_modelo_real
        )
        matriz_df = pd.DataFrame(
            matriz,
            index=df["ID"],
            columns=df["ID"]
        )
        st.dataframe(
            matriz_df.round(3),
            use_container_width=True
        )

with abas[2]:
    st.subheader("Espaço Vetorial")

    metodo = st.selectbox(
        "Redução de dimensionalidade",
        ["PCA", "t-SNE"]
    )

    embeddings = gerar_embeddings(
        tuple(df["Texto"]),
        nome_modelo_real
    )

    if metodo == "PCA":
        coordenadas = PCA(
            n_components=2,
            random_state=42
        ).fit_transform(embeddings)
    else:
        coordenadas = TSNE(
            n_components=2,
            random_state=42,
            perplexity=10,
            init="pca",
            learning_rate="auto"
        ).fit_transform(embeddings)

    visual = df.copy()
    visual["x"] = coordenadas[:, 0]
    visual["y"] = coordenadas[:, 1]

    st.scatter_chart(
        visual,
        x="x",
        y="y",
        color="Categoria",
        size=80
    )

    st.caption(
        "Os agrupamentos representam proximidade semântica no espaço reduzido. "
        "As categorias oficiais podem se sobrepor porque problemas diferentes podem "
        "compartilhar contexto e vocabulário."
    )

with abas[3]:
    st.subheader("Chunking de Manifestações")

    texto_longo = st.text_area(
        "Cole uma manifestação longa",
        height=220
    )

    estrategia = st.selectbox(
        "Estratégia",
        [
            "300 caracteres / overlap 30",
            "500 caracteres / overlap 100"
        ]
    )

    if estrategia.startswith("300"):
        chunk_size, chunk_overlap = 300, 30
    else:
        chunk_size, chunk_overlap = 500, 100

    if st.button("Gerar chunks"):
        if not texto_longo.strip():
            st.warning("Cole uma manifestação para gerar os chunks.")
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(texto_longo)

            modelo = carregar_modelo(nome_modelo_real)
            embeddings_chunks = modelo.encode(
                chunks,
                normalize_embeddings=True
            )

            st.write(
                f"**{len(chunks)} chunks** — "
                f"chunk_size={chunk_size}, overlap={chunk_overlap}"
            )

            for i, (chunk, embedding) in enumerate(
                zip(chunks, embeddings_chunks),
                start=1
            ):
                st.markdown(f"**Chunk {i}**")
                st.write(chunk)
                st.write(
                    "Embedding:",
                    np.round(embedding[:10], 4),
                    "..."
                )
                st.divider()

st.sidebar.divider()
st.sidebar.caption(
    "Modelos: Sentence Transformers. "
    "Embeddings são armazenados em cache para evitar processamento repetido."
)
