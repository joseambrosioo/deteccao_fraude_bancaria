# train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def pre_processar_dados(caminho_dados):
    dados = pd.read_csv(caminho_dados)
    dados_reduzidos = dados.drop(['zipcodeOri', 'zipMerchant'], axis=1)
    colunas_categoricas = dados_reduzidos.select_dtypes(include=['object']).columns
    for col in colunas_categoricas:
        dados_reduzidos[col] = dados_reduzidos[col].astype('category')
    dados_reduzidos[colunas_categoricas] = dados_reduzidos[colunas_categoricas].apply(lambda x: x.cat.codes)
    X = dados_reduzidos.drop(['fraud'], axis=1)
    y = dados_reduzidos['fraud']
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
    )
    return X_treino, X_teste, y_treino, y_teste, X.columns

X_treino, X_teste, y_treino, y_teste, colunas_caracteristicas = pre_processar_dados('dataset/bs140513_032310.csv')

modelos_a_treinar = {
    'Classificador K-Neighbors': KNeighborsClassifier(n_neighbors=5, p=1),
    'Classificador Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"),
    'Classificador XGBoost': XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400, objective="binary:hinge", random_state=42),
}

modelos_treinados = {}
for nome, modelo in modelos_a_treinar.items():
    print(f"Treinando {nome}...")
    modelo.fit(X_treino, y_treino.values.ravel())
    modelos_treinados[nome] = modelo
    # Salva o modelo
    joblib.dump(modelo, f'models/{nome.replace(" ", "_")}.pkl')
    print(f"Modelo {nome} salvo em models/{nome.replace(' ', '_')}.pkl")

# Você também pode querer salvar seus dados de teste para garantir a consistência
joblib.dump(X_teste, 'data/X_test.pkl')
joblib.dump(y_teste, 'data/y_test.pkl')
joblib.dump(colunas_caracteristicas, 'data/feature_columns.pkl')