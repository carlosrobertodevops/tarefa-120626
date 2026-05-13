#!/usr/bin/env python
# coding: utf-8

# # Tarefa WDBC — Auditoria de Dados Clínicos e Justiça Algorítmica
#
# **Objetivo:** auditar a base histórica **Breast Cancer Wisconsin Diagnostic — WDBC**, sem treinar modelo preditivo, respondendo aos requisitos analíticos da especificação.
#
# > Observação operacional: neste ambiente foi encontrado o arquivo `wdbc.data`. Não foi localizado um arquivo `wdbc.bat`; por isso, o notebook usa `wdbc.data` como base bruta da atividade.
#

# ## Sumário dos requisitos atendidos
#
# 1. **Estruturação da base de dados:** ingestão, nomeação das colunas, seleção dos atributos e codificação binária do diagnóstico.
# 2. **Diagnóstico de assimetria biológica:** análise estatística da variável `area_mean`, com medidas de tendência central, dispersão, assimetria e gráfico de densidade.
# 3. **Isolamento robusto de anomalias:** detecção não-paramétrica de outliers por regra de IQR, cálculo do limite de corte e prevalência de malignidade no subgrupo isolado.
# 4. **Mapeamento de multicolinearidade:** matriz de correlação entre as variáveis extraídas das imagens celulares e visualização por heatmap.
#

# In[1]:

# Requisito I — Preparação do ambiente e ingestão da base

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

# Caminhos possíveis para execução local ou no ambiente do ChatGPT
candidate_paths = [
    Path('wdbc.data'),
]

data_path = next((p for p in candidate_paths if p.exists()), None)
if data_path is None:
    raise FileNotFoundError('Arquivo wdbc.data não encontrado. Coloque-o no mesmo diretório do notebook.')

print(f'Arquivo carregado: {data_path}')
# In[2]:

# Definição oficial das colunas do WDBC
# Estrutura: ID, diagnóstico e 30 atributos numéricos derivados da imagem celular.

base_features = [
    'radius', 'texture', 'perimeter', 'area', 'smoothness',
    'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
]

columns = ['id', 'diagnosis']
for suffix in ['mean', 'se', 'worst']:
    columns.extend([f'{feature}_{suffix}' for feature in base_features])

raw_df = pd.read_csv(data_path, header=None, names=columns)
raw_df.head()

# In[3]:

# Isolamento das características morfológicas e codificação matemática da variável alvo.
# M = maligno -> 1
# B = benigno -> 0

df = raw_df.copy()
df['diagnosis_binary'] = df['diagnosis'].map({'B': 0, 'M': 1}).astype(int)

feature_columns = [c for c in df.columns if c not in ['id', 'diagnosis', 'diagnosis_binary']]
model_ready_df = df[['diagnosis_binary'] + feature_columns]

print('Volumetria final dos dados tratados')
print(f'Linhas/pacientes: {model_ready_df.shape[0]}')
print(f'Colunas finais: {model_ready_df.shape[1]}')
print(f'Atributos morfológicos: {len(feature_columns)}')
print('\nDistribuição do diagnóstico original:')
print(df['diagnosis'].value_counts().rename({'B': 'Benigno', 'M': 'Maligno'}))

model_ready_df.head()


# ### Parecer do Requisito I
#
# A base foi estruturada com **569 pacientes**, **30 atributos morfológicos/imagem-derivados** e uma variável-alvo binária (`diagnosis_binary`) compatível com algoritmos de classificação binária, embora o escopo da atividade seja **auditoria estatística**, não treinamento de modelo.
#

# In[4]:


# Requisito II — Diagnóstico de assimetria biológica da área média do núcleo celular

area = df['area_mean']

area_stats = pd.Series({
    'média': area.mean(),
    'mediana': area.median(),
    'desvio_padrão': area.std(),
    'mínimo': area.min(),
    'Q1': area.quantile(0.25),
    'Q3': area.quantile(0.75),
    'máximo': area.max(),
    'assimetria_skewness': area.skew(),
    'curtose': area.kurtosis(),
    'média_menos_mediana': area.mean() - area.median(),
    'max_sobre_mediana': area.max() / area.median(),
})

area_stats.to_frame('valor')


# In[5]:


# Prova visual obrigatória: gráfico de densidade da área média

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='area_mean', fill=True, linewidth=2)
plt.axvline(area.mean(), linestyle='--', label=f'Média = {area.mean():.2f}')
plt.axvline(area.median(), linestyle=':', label=f'Mediana = {area.median():.2f}')
plt.title('Densidade de area_mean — evidência de cauda longa à direita')
plt.xlabel('Área média do núcleo celular (area_mean)')
plt.ylabel('Densidade')
plt.legend()
plt.show()


# ### Parecer do Requisito II
#
# A variável `area_mean` apresenta **assimetria positiva relevante**: a média fica acima da mediana, o valor máximo é muitas vezes maior que a mediana e a cauda direita é visualmente extensa no gráfico de densidade. Essa morfologia estatística sugere que métodos paramétricos baseados em normalidade podem ser inadequados para definir anomalias clínicas nessa variável.
#

# In[6]:

# Requisito III — Isolamento robusto de anomalias por método não-paramétrico
# Justificativa: como area_mean é assimétrica, usamos a regra do IQR, que não pressupõe normalidade.

q1 = area.quantile(0.25)
q3 = area.quantile(0.75)
iqr = q3 - q1
upper_cutoff = q3 + 1.5 * iqr

outliers_area = df[df['area_mean'] > upper_cutoff].copy()
malignant_prevalence = outliers_area['diagnosis_binary'].mean()
outlier_rate = len(outliers_area) / len(df)

print('Limite robusto de corte para outliers superiores em area_mean')
print(f'Q1: {q1:.2f}')
print(f'Q3: {q3:.2f}')
print(f'IQR: {iqr:.2f}')
print(f'Corte superior = Q3 + 1.5*IQR: {upper_cutoff:.2f}')
print('\nSubgrupo isolado')
print(f'Pacientes atípicos por área: {len(outliers_area)} de {len(df)} ({outlier_rate:.2%})')
print(f'Taxa de prevalência de tumores malignos nesse subgrupo: {malignant_prevalence:.2%}')
print('\nDistribuição no subgrupo isolado:')
print(outliers_area['diagnosis'].value_counts())

outliers_area[['id', 'diagnosis', 'diagnosis_binary', 'area_mean']].sort_values('area_mean', ascending=False).head(10)

# In[7]:

# Visualização auxiliar dos outliers isolados por IQR

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='diagnosis', y='area_mean')
plt.axhline(upper_cutoff, linestyle='--', label=f'Corte IQR = {upper_cutoff:.2f}')
plt.title('Outliers superiores de area_mean por diagnóstico')
plt.xlabel('Diagnóstico original')
plt.ylabel('Área média do núcleo celular')
plt.legend()
plt.show()


# ### Parecer do Requisito III
#
# O limite de corte não-paramétrico por IQR isola pacientes com `area_mean` superior a `Q3 + 1.5 × IQR`. No arquivo analisado, esse subgrupo contém apenas casos malignos, produzindo uma prevalência de malignidade de **100%** entre os pacientes atípicos por área celular. Isso não significa causalidade, mas indica forte associação clínica entre áreas nucleares extremas e diagnóstico maligno na amostra.
#

# In[8]:

# Requisito IV — Mapeamento de multicolinearidade

corr_matrix = df[feature_columns].corr(method='pearson')

plt.figure(figsize=(18, 14))
sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.2,
    cbar_kws={'label': 'Correlação de Pearson'}
)
plt.title('Matriz de correlação entre atributos morfológicos do WDBC')
plt.tight_layout()
plt.show()


# In[9]:

# Pares mais correlacionados em valor absoluto, excluindo a diagonal

mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
top_correlations = (
    corr_matrix.where(mask)
    .stack()
    .rename('correlacao')
    .reset_index()
    .rename(columns={'level_0': 'variavel_1', 'level_1': 'variavel_2'})
)
top_correlations['correlacao_abs'] = top_correlations['correlacao'].abs()
top_correlations = top_correlations.sort_values('correlacao_abs', ascending=False)

top_correlations.head(15)


# ### Parecer do Requisito IV
#
# A matriz de correlação evidencia **multicolinearidade forte** entre variáveis geométricas relacionadas, especialmente `radius`, `perimeter` e `area` em suas versões `mean`, `se` e `worst`. Isso indica redundância informacional: em um eventual sistema de triagem, incluir todas essas variáveis sem controle pode aumentar instabilidade interpretativa e super-representar o mesmo fenômeno geométrico.
#

# ## Parecer final — viabilidade e riscos éticos
#
# A base WDBC é estatisticamente útil para auditoria e estudo exploratório, mas exige cautela antes de uso em sistemas de triagem na rede pública:
#
# - **Viabilidade técnica:** há forte separação estatística em algumas variáveis, como `area_mean`, especialmente nos extremos.
# - **Risco de viés amostral:** a base é histórica, limitada e pode não representar integralmente populações atendidas na rede pública brasileira.
# - **Risco de redundância:** a multicolinearidade pode induzir modelos a sobrevalorizar atributos geométricos correlacionados.
# - **Risco clínico:** outliers associados a malignidade devem apoiar triagem e priorização, não substituir avaliação médica.
# - **Recomendação:** antes de uso operacional, realizar validação externa, análise de subgrupos, calibração clínica, documentação de limitações e governança algorítmica.
#
# **Conclusão:** o conjunto é adequado para fins educacionais, auditoria estatística e prototipação controlada. Para triagem pública real, ele não deve ser usado isoladamente sem validação clínica, avaliação de equidade e monitoramento contínuo.
#
