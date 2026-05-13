import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Configuração visual padrão para gráficos acadêmicos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Ambiente configurado com sucesso.")

# Carregamento direto do repositório da UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Dicionário de colunas baseado na documentação oficial
colunas = [
    'idade', 'sexo', 'tipo_dor_peito', 'pressao_repouso',
    'colesterol', 'glicemia_jejum', 'ecg_repouso', 'freq_cardiaca_max',
    'angina_exercicio', 'depressao_st', 'inclinacao_st',
    'num_vasos_coloridos', 'talassemia', 'diagnostico'
]

# Lendo o arquivo e convertendo '?' para NaN (Not a Number) nativo do Pandas
df = pd.read_csv(url, names=colunas, na_values='?')

print(f"Dimensões do Dataset: {df.shape[0]} pacientes e {df.shape[1]} atributos clínicos.")

# Verificando dados nulos (Falta de dados sensíveis)
print("\nValores Nulos Detectados:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Decisão Técnica: Como a quantidade de nulos é baixa (menos de 2%), optaremos pela exclusão
# para não introduzir vieses artificiais de imputação em exames críticos como 'talassemia'.
df = df.dropna()

df.describe()

df.info()

# Variáveis clínicas contínuas
vars_continuas = ['idade', 'pressao_repouso', 'colesterol', 'freq_cardiaca_max']

# Criando um DataFrame comparativo
analise_dist = pd.DataFrame({
    'Média': df[vars_continuas].mean(),
    'Mediana': df[vars_continuas].median(),
    'Desvio Padrão': df[vars_continuas].std(),
    'Assimetria (Skewness)': df[vars_continuas].skew(),
    'Curtose (Kurtosis)': df[vars_continuas].kurt()
})

display(analise_dist)

# Visualizando a assimetria do Colesterol
plt.figure(figsize=(10, 5))
sns.histplot(df['colesterol'], kde=True, color='darkblue')
plt.title('Distribuição de Colesterol Sérico')
plt.xlabel('Colesterol (mg/dl)')
plt.ylabel('Frequência de Pacientes')
plt.axvline(df['colesterol'].mean(), color='red', linestyle='--', label='Média')
plt.axvline(df['colesterol'].median(), color='green', linestyle='-', label='Mediana')
plt.legend()
plt.show()

# Calculando a matriz de correlação (Apenas colunas numéricas)
matriz_corr = df.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap='RdBu_r', vmin=-1, vmax=1, square=True)
plt.title('Matriz de Correlação de Pearson - Identificação de Multicolinearidade', fontsize=14, pad=20)
plt.show()

# Definindo a coluna alvo para a análise
alvo = 'colesterol'

# --- Método 1: Z-Score (Distância em Desvios Padrões) ---
media = df[alvo].mean()
std = df[alvo].std()
df['z_score'] = np.abs((df[alvo] - media) / std)

# Pacientes com Z-Score > 3
outliers_z = df[df['z_score'] > 3]

# --- Método 2: IQR (Intervalo Interquartil - Robusto a assimetria) ---
Q1 = df[alvo].quantile(0.25)
Q3 = df[alvo].quantile(0.75)
IQR = Q3 - Q1
limite_sup = Q3 + 1.5 * IQR

outliers_iqr = df[df[alvo] > limite_sup]

print(f"Anomalias isoladas pelo Z-Score (Z > 3): {len(outliers_z)} pacientes.")
print(f"Anomalias isoladas pelo IQR (Robustez): {len(outliers_iqr)} pacientes.\n")

#BPLOT
plt.figure(figsize=(12, 5))

# Desenha o Boxplot e os pontos dos pacientes
sns.boxplot(x=df[alvo], color='lightgray')
sns.stripplot(x=df[alvo], color='red', alpha=0.5, jitter=True)

# 1. Linha do Limite IQR (Robusto)
plt.axvline(limite_sup, color='black', linestyle='--', linewidth=2,
            label=f'Limite Superior IQR ({limite_sup:.1f})')

# 2. Linha do Limite Z-Score (Paramétrico)
limite_zscore = media + (3 * std)
plt.axvline(limite_zscore, color='blue', linestyle='-.', linewidth=2,
            label=f'Limite Z-Score > 3 ({limite_zscore:.1f})')

plt.title('Comparação de Detecção de Anomalias: IQR vs Z-Score')
plt.xlabel('Nível de Colesterol (mg/dl)')
plt.legend()
plt.show()

# Binarizando o diagnóstico (0 = saudável, >0 = presença de doença cardíaca)
df['doenca_cardiaca'] = df['diagnostico'].apply(lambda x: 1 if x > 0 else 0)

# Mapeando o sexo para legibilidade (Conforme dicionário da UCI)
df['genero'] = df['sexo'].map({1.0: 'Masculino', 0.0: 'Feminino'})

print("Contagem", df['genero'].value_counts())
print("Contagem de doentes por sexo", df['genero'].where(df['doenca_cardiaca'] == 1).value_counts())

# Calculando a taxa de diagnóstico positivo por gênero (Demographic Parity)
taxa_diagnostico = df.groupby('genero')['doenca_cardiaca'].mean() * 100

print("--- Auditoria de Paridade Demográfica (Historical Data) ---")
print(f"Taxa de diagnóstico positivo em Mulheres: {taxa_diagnostico['Feminino']:.2f}%")
print(f"Taxa de diagnóstico positivo em Homens: {taxa_diagnostico['Masculino']:.2f}%")

# Gráfico para Evidenciar o Viés de Representação
plt.figure(figsize=(8, 5))
sns.barplot(x=taxa_diagnostico.index, y=taxa_diagnostico.values, palette='viridis')
plt.title('Disparidade Demográfica Histórica no Dataset')
plt.ylabel('Taxa de Diagnóstico Positivo (%)')
plt.ylim(0, 100)
plt.show()
