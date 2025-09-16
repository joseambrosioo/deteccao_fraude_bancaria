import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import dash_table
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --- Carregamento de Dados (Carrega todos os dados e modelos UMA VEZ na inicialização) ---
try:
    dados = pd.read_csv('dataset/bs140513_032310.csv')
    X_teste_bruto = joblib.load('data/X_test.pkl')
    y_teste = joblib.load('data/y_test.pkl')
    colunas_caracteristicas = joblib.load('data/feature_columns.pkl')
    X_teste = pd.DataFrame(X_teste_bruto, columns=colunas_caracteristicas)
    
    # CARREGA TODOS OS MODELOS E OS ARMAZENA NA MEMÓRIA
    resultados_modelo = {
        'Classificador K-Neighbors': joblib.load('models/Classificador_K-Neighbors.pkl'),
        'Classificador Random Forest': joblib.load('models/Classificador_Random_Forest.pkl'),
        'Classificador XGBoost': joblib.load('models/Classificador_XGBoost.pkl'),
    }
except FileNotFoundError as e:
    print(f"Arquivos de modelo ou dados não encontrados. Por favor, execute o script de treinamento primeiro. Erro: {e}")
    exit()

y = dados['fraud']

# --- Layout do Dashboard ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Detecção de Fraude em Pagamentos Bancários"
server = app.server

cabecalho = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("💰", className="me-2"),
                    dbc.NavbarBrand("Detecção de Fraude em Pagamentos Bancários", class_name="fw-bold text-wrap", style={"color": "black"}),
                ], className="d-flex align-items-center"
            ),
            dbc.Badge("Dashboard", color="primary", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# 1. Aba ASK (PERGUNTAR)
aba_perguntar = dcc.Markdown(
    """
    ### ❓ **PERGUNTAR** — A Visão Geral

    Esta seção define o problema central e o valor de negócio deste projeto.

    **Tarefa de Negócio**: O objetivo principal é construir um sistema inteligente que possa identificar com precisão transações bancárias fraudulentas do **dataset Banksim**. A fraude é um problema multibilionário que afeta tanto as instituições financeiras quanto seus clientes. Ao detectar e prevenir essas transações fraudulentas em tempo real, podemos minimizar perdas financeiras, proteger clientes e manter a confiança.

    **Partes Interessadas**: Os principais usuários deste dashboard seriam **Analistas de Fraude**, **Equipes de Gestão de Risco** e a **Liderança Executiva**. Eles precisam de uma visão clara e fácil de entender do desempenho do modelo e das características da atividade fraudulenta para tomar decisões informadas e implantar estratégias eficazes.

    **Entregas**: O produto final é este dashboard interativo, que apresenta uma análise abrangente, exibe o desempenho de vários modelos de aprendizado de máquina e oferece insights acionáveis para melhorar a detecção de fraudes.
    """, className="p-4"
)

# 2. Aba PREPARE (PREPARAR)
aba_preparar = html.Div(
    children=[
        html.H4(["📝 ", html.B("PREPARAR"), " — Preparando os Dados"], className="mt-4"),
        html.P("Antes de construir um modelo preditivo, precisamos entender e preparar nossos dados."),
        html.H5("Fonte de Dados"),
        html.P([
            "Estamos usando o ",
            html.B("dataset Banksim"),
            ", um dataset gerado sinteticamente que simula pagamentos bancários. Ele contém quase 600.000 transações com várias características."
        ]),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Resumo do Dataset"),
                            dbc.CardBody(
                                [
                                    html.P(f"Total de Transações: {dados.shape[0]}"),
                                    html.P(f"Características: {dados.shape[1]}"),
                                    html.P(f"Transações Normais: {y.value_counts()[0]}"),
                                    html.P(f"Transações Fraudulentas: {y.value_counts()[1]}"),
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Problema de Dados Não Balanceados"),
                            dbc.CardBody(
                                [
                                    html.P([
                                        "Como é comum em dados de fraude, o dataset é altamente ",
                                        html.B("não balanceado"),
                                        ". Apenas uma pequena fração de todas as transações é fraudulenta."
                                    ]),
                                    html.P([
                                        "Para resolver isso, usamos uma técnica chamada ",
                                        html.B("SMOTE (Synthetic Minority Over-sampling Technique)"),
                                        ". Em vez de apenas copiar os exemplos de fraude, o SMOTE cria de forma inteligente novos pontos de dados de fraude sintéticos que são semelhantes aos existentes, ajudando nossos modelos a aprender de forma mais eficaz."
                                    ]),
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
            ]
        ),
        html.H5("Descrições das Características"),
        dcc.Markdown(
            """
            O dataset inclui as seguintes características principais:
            - **Step**: O dia da simulação, de 0 a 180 (6 meses).
            - **Customer** e **Merchant**: IDs anonimizados para o cliente e o comerciante.
            - **Age** e **Gender**: Informações demográficas, categorizadas em grupos.
            - **Category**: O tipo de compra (por exemplo, 'es_travel', 'es_health').
            - **Amount**: O valor da transação.
            - **Fraud**: Nossa variável-alvo. 1 significa fraudulenta, 0 significa não fraudulenta.
            """, className="p-4"
        ),
        html.H5("Amostra do Dataset (Primeiras 10 Linhas)"),
        dash_table.DataTable(
            id='tabela',
            columns=[
                {"name": "ageGroup" if i == 'age' else i, "id": i, "type": "numeric" if i in ['step', 'amount', 'fraud'] else "text"}
                for i in dados.columns
            ],
            data=dados.head(10).to_dict('records'),
            sort_action="native",
            filter_action="native",
            page_action="none",
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'textAlign': 'center',
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'font-size': '12px',
                'minWidth': '80px', 'width': 'auto', 'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
        ),
    ], className="p-4"
)

# 3. Aba ANALYZE (ANALISAR) com sub-abas
aba_analisar = html.Div(
    children=[
        html.H4(["📈 ", html.B("ANALISAR"), " — Encontrando Padrões e Construindo Modelos"], className="mt-4"),
        html.P("É aqui que exploramos os dados para encontrar padrões e construir o cérebro preditivo do nosso dashboard."),
        dbc.Tabs([
            dbc.Tab(label="Análise Exploratória de Dados", children=[
                html.Div(
                    children=[
                        html.H5("Transações Fraudulentas vs. Não Fraudulentas", className="mt-4"),
                        html.P(
                            """
                            Um dos insights mais significativos de nossa exploração de dados é a diferença nos valores das transações. Transações fraudulentas tendem a ter um valor médio muito maior do que as não fraudulentas. Os fraudadores muitas vezes buscam alvos de alto valor.
                            """
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="boxplot-valor")),
                        ]),
                        html.P([
                            html.B("Insight do Box Plot:"),
                            " O box plot mostra a distribuição dos valores das transações em diferentes categorias de compra. Embora a maioria das categorias tenha uma faixa de valor semelhante, a categoria ",
                            html.B("'es_travel'"),
                            " se destaca com valores de transação extremamente altos. Isso sugere que os fraudadores visam categorias onde transações de alto valor são comuns, tornando sua atividade menos suspeita."
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="histograma-valor")),
                        ]),
                        html.P([
                            html.B("Insight do Histograma:"),
                            " Este gráfico mostra claramente o desequilíbrio nos dados. Há muito menos transações fraudulentas, mas elas estão concentradas em valores muito mais altos, enquanto as transações benignas são de baixo valor e muito frequentes. Este é um padrão clássico na detecção de fraudes e confirma nossa hipótese."
                        ]),
                        html.H5("Fraude por Categoria e Grupo Etário", className="mt-4"),
                        html.P("Também exploramos como a fraude se distribui por diferentes categorias de compra e grupos etários."),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="barra-fraude-categoria"), md=6),
                            dbc.Col(dcc.Graph(id="barra-fraude-idade"), md=6),
                        ]),
                        html.P([
                            html.B("Insight da Fraude por Categoria:"),
                            " O gráfico de barras mostra a porcentagem de transações fraudulentas dentro de cada categoria. ",
                            html.B("'es_leisure'"),
                            " e ",
                            html.B("'es_travel'"),
                            " têm as maiores taxas de fraude, reforçando a ideia de que os fraudadores visam categorias de alto valor e gastos discricionários."
                        ]),
                        html.H5("Análise por Categoria"),
                        dcc.Markdown(
                            """
                            O dataset inclui as seguintes categorias:
                            - **es_leisure**: Representa transações para atividades recreativas, como entretenimento, hobbies ou bens de luxo.
                            - **es_travel**: Transações relacionadas a viagens, incluindo voos, hotéis e transporte.
                            - **es_health**: Transações para serviços e produtos médicos ou de saúde.
                            - **es_hotelservices**: Transações para estadias em hotéis e serviços relacionados.
                            - **es_barsandrestaurants**: Pagamentos feitos em bares e restaurantes.
                            - **es_transportation**: Transações para transporte, como passagens de ônibus ou trem.
                            - **es_sportsandoutdoors**: Compras de equipamentos esportivos ou artigos para atividades ao ar livre.
                            - **es_contents**: Transações para conteúdo digital ou mídia.
                            - **es_fashion**: Compras de roupas e acessórios.
                            - **es_tech**: Transações para eletrônicos e tecnologia.
                            - **es_home**: Compras relacionadas a produtos e serviços para casa.
                            - **es_shopping_net**: Transações de compras online.
                            - **es_others**: Uma categoria genérica para transações que não se encaixam nas outras.
                            - **es_food**: Transações para supermercado ou itens alimentícios.
                            - **es_service**: Transações para serviços em geral.
                            - **es_shopping**: Transações de compras presenciais.
                            """, className="p-4"
                        ),
                        html.P([
                            html.B("Insight da Fraude por Grupo Etário:"),
                            " Curiosamente, o grupo etário com menos de 18 anos (categoria '0') tem a maior porcentagem de fraude. Isso pode ser devido a uma série de razões, como indivíduos mais jovens sendo mais suscetíveis a roubo de identidade ou fraudadores intencionalmente usando perfis de idade mais jovem."
                        ]),
                        html.H5("Análise por Grupo Etário"),
                        dcc.Markdown(
                            """
                            - **Grupo Etário 0**: Menos de 18 anos. Este grupo mostra a maior taxa de fraude, o que pode ser um resultado de comportamento online menos seguro ou do uso de credenciais roubadas em contas com medidas de segurança menos rigorosas.
                            - **Grupo Etário 1**: 18-25 anos.
                            - **Grupo Etário 2**: 26-35 anos.
                            - **Grupo Etário 3**: 36-45 anos.
                            - **Grupo Etário 4**: 46-55 anos.
                            - **Grupo Etário 5**: 56-65 anos.
                            - **Grupo Etário 6**: Acima de 65 anos.
                            """, className="p-4"
                        ),
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Desempenho do Modelo", children=[
                html.Div(
                    children=[
                        html.H5("Desempenho do Modelo em Dados de Teste", className="mt-4"),
                        html.P([
                            "Treinamos três modelos diferentes de aprendizado de máquina: ",
                            html.B("K-Nearest Neighbors (KNN)"),
                            ", ",
                            html.B("Random Forest"),
                            " e ",
                            html.B("XGBoost"),
                            ". Esses modelos são avaliados em um conjunto de 'teste' separado para garantir que não estejam apenas memorizando os dados de treinamento."
                        ]),
                        html.P(
                            ["Para realmente avaliar nossos modelos de detecção de fraude, focamos em várias métricas-chave além da simples acurácia:",
                            html.Ul([
                                html.Li([html.B("Precisão:"), " Pense na Precisão como o custo de um alarme falso. Se nosso modelo sinaliza uma transação como fraudulenta, alta precisão significa que é muito provável que ela seja realmente fraudulenta. De todas as transações que nosso modelo sinalizou, quantas eram realmente fraudulentas? Alta precisão é boa para reduzir investigações desnecessárias."]),
                                html.Li([html.B("Recall (Sensibilidade):"), " Pense no Recall como o custo de uma fraude perdida. Alto recall significa que nosso modelo captura a maioria das transações fraudulentas reais, para que não deixemos a fraude passar despercebida. De todas as transações fraudulentas, quantas nosso modelo identificou com sucesso? Alto recall é crucial para evitar perdas financeiras."]),
                                html.Li([html.B("F1-Score:"), " Este é um equilíbrio entre precisão e recall, fornecendo uma única métrica para comparar modelos. É a média harmônica da precisão e do recall, resumindo tanto alarmes falsos quanto fraudes perdidas em um único número."]),
                                html.Li([html.B("ROC-AUC:"), " Esta é uma poderosa métrica de resumo que mede a capacidade do modelo de distinguir entre transações fraudulentas e não fraudulentas. Uma pontuação mais próxima de 1.0 indica que o modelo pode separar de forma confiável as duas classes, tornando-o altamente eficaz para a tomada de decisões."])
                            ])
                            ]
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="barra-metricas-modelo"), md=12),
                        ]),
                        
                        # Nova seção de texto para descrever o desempenho do modelo com números
                        html.P([
                            "Nossa análise mostra que o ", html.B("Classificador XGBoost"), " teve um desempenho excepcionalmente bom, alcançando uma ", html.B("Precisão de 0.99"), ", um ", html.B("Recall de 0.99"), ", um ", html.B("F1-Score de 0.99"), ", e um ", html.B("ROC-AUC de 0.99"),
                            ". Esses resultados, embora incrivelmente fortes, devem ser considerados juntamente com a matriz de confusão completa para uma visão completa do desempenho do modelo. O ", html.B("Classificador K-Neighbors"), " também teve um desempenho excepcionalmente bom, com uma ", html.B("Precisão de 0.98"), ", um ", html.B("Recall de 0.99"), ", um ", html.B("F1-Score de 0.99"), ", e um ", html.B("ROC-AUC de 0.99"),
                            ". O ", html.B("Classificador Random Forest"), " teve um desempenho menor, mas ainda forte, com uma ", html.B("Precisão de 0.97"), ", um ", html.B("Recall de 0.99"), ", um ", html.B("F1-Score de 0.98"), ", e um ", html.B("ROC-AUC de 0.99"), ". O desempenho superior do XGBoost em algumas métricas-chave o torna um forte candidato, mas os outros modelos também apresentam resultados convincentes."
                        ]),
                        html.Hr(),
                        html.H5("Matriz de Confusão e Curva ROC", className="mt-4"),
                        html.P("Selecione um modelo para visualizar sua matriz de confusão e curva ROC específicas:"),
                        dcc.Dropdown(
                            id='seletor-modelo-dropdown',
                            options=[{'label': i, 'value': i} for i in resultados_modelo.keys()],
                            value='Classificador XGBoost',
                            clearable=False,
                            style={'width': '50%', 'margin-bottom': '20px'}
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="matriz-confusao"), md=6),
                            dbc.Col(dcc.Graph(id="curva-roc"), md=6),
                        ]),
                        html.H6("Matriz de Confusão", className="mt-4"),
                        html.P(
                            ["A matriz de confusão é uma tabela que divide as previsões do nosso modelo em quatro categorias:",
                            html.Ul([
                                html.Li([html.B("Verdadeiros Positivos (VP):"), " Pagamentos fraudulentos corretamente previstos."]),
                                html.Li([html.B("Verdadeiros Negativos (VN):"), " Pagamentos legítimos corretamente previstos."]),
                                html.Li([html.B("Falsos Positivos (FP):"), " Pagamentos fraudulentos incorretamente previstos (erro Tipo I). São transações legítimas sinalizadas como fraude, o que pode ser um inconveniente para os clientes."]),
                                html.Li([html.B("Falsos Negativos (FN):"), " Pagamentos legítimos incorretamente previstos (erro Tipo II). São transações fraudulentas que foram perdidas pelo modelo, representando uma perda financeira para o banco e um risco de segurança para o cliente."])
                            ])
                            ]
                        ),
                        html.P([
                            "Para fornecer uma visão mais granular do desempenho de cada modelo, podemos olhar para os resultados da ",
                            html.B("matriz de confusão"), ". O ",
                            html.B("Classificador XGBoost"), " teve uma acurácia de ",
                            html.B("99.15%"), " com ",
                            html.B("175.490 verdadeiros positivos (VP)"), " e ",
                            html.B("173.997 verdadeiros negativos (VN)"), ", enquanto classificou incorretamente apenas ",
                            html.B("2.236 transações como falsos positivos (FP)"), " e ",
                            html.B("743 como falsos negativos (FN)"), ". O ",
                            html.B("Classificador K-Neighbors"), " teve uma acurácia de ",
                            html.B("98.70%"), " com ",
                            html.B("175.871 verdadeiros positivos (VP)"), " e ",
                            html.B("171.999 verdadeiros negativos (VN)"), ", com ",
                            html.B("4.234 falsos positivos (FP)"), " e ",
                            html.B("362 falsos negativos (FN)"), ". Por fim, o ",
                            html.B("Classificador Random Forest"), " teve uma acurácia de ",
                            html.B("97.96%"), " ao identificar corretamente ",
                            html.B("175.154 verdadeiros positivos (VP)"), " e ",
                            html.B("170.106 verdadeiros negativos (VN)"), ", com ",
                            html.B("6.127 falsos positivos (FP)"), " e ",
                            html.B("1.079 falsos negativos (FN)"), ". Esses números ressaltam o excelente equilíbrio que cada modelo alcança entre capturar fraudes e evitar alarmes falsos."
                        ]),
                        html.H6("Curva Característica de Operação do Receptor (ROC)", className="mt-4"),
                        html.P([
                            "A curva ROC plota a ",
                            html.B("Taxa de Verdadeiros Positivos"),
                            " contra a ",
                            html.B("Taxa de Falsos Positivos"),
                            ". Quanto mais próxima a curva estiver do canto superior esquerdo, melhor o modelo distingue entre as duas classes (fraude e não fraude). A Área sob a Curva (AUC) fornece uma única métrica para resumir o desempenho do modelo.",
                        ]),
                        html.P([
                            "O ",
                            html.B("Classificador Random Forest"),
                            " e o ",
                            html.B("Classificador K-Neighbors"),
                            " ambos alcançaram uma curva ROC perfeita com um ",
                            html.B("AUC de 1.00"),
                            ", enquanto o ",
                            html.B("Classificador XGBoost"),
                            " ficou muito próximo com um ",
                            html.B("AUC de 0.99"),
                            ". Esses resultados demonstram a excelente capacidade dos modelos de diferenciar entre transações fraudulentas e não fraudulentas. Todos os três modelos são altamente eficazes na identificação de fraudes, com os classificadores Random Forest e K-Neighbors performando um pouco melhor nesta métrica específica."
                        ]),
                        html.Hr(),
                        html.H5("Importância das Características (para modelos baseados em árvores)", className="mt-4"),
                        html.P("Este gráfico classifica as características com base em sua contribuição para a previsão do modelo."),
                        dcc.Dropdown(
                            id="dropdown-importancia-caracteristica",
                            options=[
                                {'label': 'Classificador Random Forest', 'value': 'Classificador Random Forest'},
                                {'label': 'Classificador XGBoost', 'value': 'Classificador XGBoost'}
                            ],
                            value='Classificador XGBoost'
                        ),
                        dcc.Graph(id="grafico-importancia-caracteristica"),
                    ], className="p-4"
                )
            ]),
        ])
    ]
)

# 4. Aba ACT (AGIR)
aba_agir = dcc.Markdown(
    """
    ### 🚀 **AGIR** — O Que Fazer a Seguir

    Esta é a seção mais importante, pois traduz nossos insights de dados em uma estratégia de negócio.

    -   **Implantar o Melhor Modelo**: O **Classificador XGBoost** é nosso modelo recomendado para implantação devido ao seu desempenho superior nos dados de teste. Este modelo será o cérebro por trás de nosso novo sistema proativo de detecção de fraudes.
    -   **Alertas em Tempo Real**: O modelo implantado deve ser usado para fornecer pontuações de risco em tempo real para cada transação. Qualquer transação com uma alta pontuação de fraude pode ser automaticamente sinalizada para revisão ou instantaneamente recusada.
    -   **Criação de Regras Direcionadas**: Nossa análise revelou que as transações fraudulentas estão frequentemente ligadas a **categorias específicas (como 'es_leisure' e 'es_travel')** e têm **valores altos**. Esses insights podem ser usados para criar regras de negócio adicionais e mais específicas que trabalham em conjunto com o modelo de aprendizado de máquina, criando uma defesa mais robusta contra fraudes.
    -   **Melhoria Contínua**: O desempenho do modelo deve ser monitorado ao longo do tempo. À medida que novos padrões de fraude surgem, o modelo deve ser retreinado com novos dados para garantir que continue eficaz.
    """, className="p-4"
)

app.layout = dbc.Container(
    [
        cabecalho,
        dbc.Tabs(
            [
                dbc.Tab(aba_perguntar, label="Perguntar"),
                dbc.Tab(aba_preparar, label="Preparar"),
                dbc.Tab(aba_analisar, label="Analisar"),
                dbc.Tab(aba_agir, label="Agir"),
            ]
        ),
    ],
    fluid=True,
)

# --- Callbacks para Gráficos ---
@app.callback(
    Output("boxplot-valor", "figure"),
    Input("histograma-valor", "id") # Input fictício para acionar na inicialização
)
def atualizar_boxplot_valor(dummy):
    fig = go.Figure()
    for categoria in dados['category'].unique():
        df_cat = dados[dados['category'] == categoria]
        fig.add_trace(go.Box(
            y=df_cat['amount'],
            name=categoria,
        ))
    
    fig.update_layout(
        title="Valor da Transação por Categoria",
        yaxis_title="Valor",
        showlegend=False,
        height=600,
        margin=dict(t=50, b=50),
    )
    fig.update_yaxes(range=[0, 1000]) # Define o limite do eixo y para focar na distribuição principal
    return fig

@app.callback(
    Output("histograma-valor", "figure"),
    Input("boxplot-valor", "id") # Input fictício para acionar na inicialização
)
def atualizar_histograma_valor(dummy):
    df_fraude = dados[dados['fraud'] == 1]
    df_nao_fraude = dados[dados['fraud'] == 0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_fraude['amount'], name='Fraudulenta', marker_color='red'))
    fig.add_trace(go.Histogram(x=df_nao_fraude['amount'], name='Não Fraudulenta', marker_color='blue'))
    fig.update_layout(
        title="Distribuição dos Valores de Transação",
        xaxis_title="Valor",
        yaxis_title="Contagem",
        barmode='overlay',
        bargap=0.2,
        height=600,
        margin=dict(t=50, b=50)
    )
    fig.update_traces(opacity=0.75)
    fig.update_xaxes(range=[0, 2000])
    return fig

@app.callback(
    Output("barra-fraude-categoria", "figure"),
    Input("barra-fraude-idade", "id") # Input fictício para acionar na inicialização
)
def atualizar_barra_fraude_categoria(dummy):
    fraude_por_categoria = dados.groupby('category')['fraud'].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=fraude_por_categoria['category'],
        y=fraude_por_categoria['fraud'] * 100,
        marker_color='lightblue'
    ))
    fig.update_layout(
        title="Porcentagem de Transações Fraudulentas por Categoria",
        xaxis_title="Categoria",
        yaxis_title="Porcentagem de Fraude (%)",
        height=500,
        margin=dict(l=50, r=50, t=50, b=150)
    )
    return fig

@app.callback(
    Output("barra-fraude-idade", "figure"),
    Input("barra-fraude-categoria", "id") # Input fictício para acionar na inicialização
)
def atualizar_barra_fraude_idade(dummy):
    fraude_por_idade = dados.groupby('age')['fraud'].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=fraude_por_idade['age'],
        y=fraude_por_idade['fraud'] * 100,
        marker_color='lightgreen'
    ))
    fig.update_layout(
        title="Porcentagem de Transações Fraudulentas por Grupo Etário",
        xaxis_title="Grupo Etário",
        yaxis_title="Porcentagem de Fraude (%)",
        height=500,
        margin=dict(t=50, b=50)
    )
    return fig

@app.callback(
    Output("barra-metricas-modelo", "figure"),
    # O Input para este callback não é necessário, pois é estático na inicialização.
    # Usamos um input fictício para garantir que ele seja executado na inicialização.
    Input('barra-fraude-idade', 'id')
)
def atualizar_barra_metricas_modelo(dummy):
    # Gráfico de Barras de Métricas do Modelo
    linhas_df = []
    for nome, modelo in resultados_modelo.items():
        previsoes = modelo.predict(X_teste)
        
        if hasattr(modelo, 'predict_proba'):
            probabilidades = modelo.predict_proba(X_teste)[:, 1]
        else:
            probabilidades = modelo.decision_function(X_teste)

        linhas_df.append({
            'Modelo': nome,
            'Precisão': precision_score(y_teste, previsoes, zero_division=0),
            'Recall': recall_score(y_teste, previsoes, zero_division=0),
            'F1-Score': f1_score(y_teste, previsoes, zero_division=0),
            'ROC-AUC': roc_auc_score(y_teste, probabilidades),
        })
    metricas_df = pd.DataFrame(linhas_df).round(4)
    
    barra_metricas = go.Figure()
    for metrica in ['Precisão', 'Recall', 'F1-Score', 'ROC-AUC']:
        barra_metricas.add_trace(go.Bar(
            y=metricas_df["Modelo"],
            x=metricas_df[metrica],
            orientation='h',
            name=metrica
        ))
    barra_metricas.update_layout(
        barmode='group',
        title="Métricas de Desempenho do Modelo",
        height=450,
        margin=dict(l=150, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return barra_metricas

@app.callback(
    Output("matriz-confusao", "figure"),
    Output("curva-roc", "figure"),
    Input('seletor-modelo-dropdown', 'value')
)
def atualizar_matriz_confusao_roc(modelo_selecionado):
    modelo = resultados_modelo[modelo_selecionado]
    y_previsao = modelo.predict(X_teste)
    
    # Obtém a matriz de confusão bruta do scikit-learn
    # [[VN, FP],
    #  [FN, VP]]
    cm = confusion_matrix(y_teste, y_previsao)
    
    # Extrai os valores explicitamente para evitar confusões
    vn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    vp = cm[1, 1]
    
    # Reordena a matriz para exibir como [[VP, FN], [FP, VN]]
    dados_z = np.array([[vp, fn],
                        [fp, vn]])

    # Cria as anotações de texto para cada célula com rótulos
    texto_cm = np.array([
        [f'VP: {vp}', f'FN: {fn}'],
        [f'FP: {fp}', f'VN: {vn}']
    ])
    
    # Cria o mapa de calor anotado
    fig_cm = ff.create_annotated_heatmap(
        z=dados_z,
        x=["Fraude Prevista (1)", "Não Fraude Prevista (0)"],
        y=["Fraude Real (1)", "Não Fraude Real (0)"],
        annotation_text=texto_cm,
        colorscale='blues',
        showscale=False
    )

    # Inverte o eixo y para que a linha superior seja "Fraude Real"
    fig_cm.update_yaxes(autorange='reversed')

    fig_cm.update_layout(
        title=f"Matriz de Confusão ({modelo_selecionado})",
        xaxis_title="Classe Prevista",
        yaxis_title="Classe Real",
        height=450,
        margin=dict(t=50, b=50)
    )
    
    # Atualiza o tamanho da fonte das anotações
    fig_cm.update_annotations(font_size=16)
    
    # Curva ROC
    fig_roc = go.Figure()
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(X_teste)[:, 1]
    else:
        probabilidades = modelo.decision_function(X_teste)

    fpr, tpr, _ = roc_curve(y_teste, probabilidades)
    roc_auc = auc(fpr, tpr)

    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Curva ROC (AUC={roc_auc:.2f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Adivinhação Aleatória'))
    fig_roc.update_layout(
        title=f"Curva ROC ({modelo_selecionado})",
        xaxis_title="Taxa de Falsos Positivos",
        yaxis_title="Taxa de Verdadeiros Positivos",
        height=450,
        margin=dict(t=50, b=50)
    )
    
    return fig_cm, fig_roc

@app.callback(
    Output("grafico-importancia-caracteristica", "figure"),
    Input("dropdown-importancia-caracteristica", "value")
)
def atualizar_importancia_caracteristica(modelo_selecionado):
    modelo = resultados_modelo[modelo_selecionado]
    
    if hasattr(modelo, 'feature_importances_'):
        colunas_caracteristicas = X_teste.columns
        importancias = modelo.feature_importances_
        df_importancia = pd.DataFrame({
            'caracteristica': colunas_caracteristicas,
            'importancia': importancias
        }).sort_values(by='importancia', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=df_importancia['importancia'],
            y=df_importancia['caracteristica'],
            orientation='h'
        ))
        fig.update_layout(
            title=f"Importância das Características para {modelo_selecionado}",
            xaxis_title="Importância",
            yaxis_title="Característica",
            height=500,
            margin=dict(l=150, t=50, b=50)
        )
        return fig
    else:
        fig = go.Figure(go.Scatter())
        fig.update_layout(title=f"Não há Importância de Características para {modelo_selecionado}", height=450, margin=dict(t=50, b=50))
        return fig

if __name__ == "__main__":
    app.run(debug=True)