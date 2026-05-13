#!/usr/bin/env python
# coding: utf-8

# # Atividade: Auditoria de Dados Clinicos e Justica Algoritmica
# 
# Notebook baseado na atividade sobre a base **Breast Cancer Wisconsin (Diagnostic)** da UCI.
# 
# Objetivo: auditar a saude estatistica dos dados, identificar anomalias clinicas e discutir riscos de uso em triagem na rede publica de saude.
# 
# Este notebook **nao treina modelo preditivo**. Ele foca em ingestao, diagnostico estatistico, outliers, multicolinearidade e parecer etico.

# ## Requisito I: Estruturacao da Base de Dados
# 
# A base bruta `wdbc.data` nao possui cabecalho. Cada linha tem:
# 
# - `id`: identificador do paciente/amostra.
# - `diagnosis`: diagnostico original (`M` = maligno, `B` = benigno).
# - 30 caracteristicas morfologicas do nucleo celular: media, erro padrao e pior valor para 10 medidas.
# 
# A variavel alvo sera codificada como:
# 
# - `M` -> `1`
# - `B` -> `0`

# In[ ]:


# Preparacao do ambiente. Se alguma biblioteca estiver ausente, ela sera instalada no kernel atual.
import importlib.util
import subprocess
import sys

required_packages = ['numpy', 'pandas', 'matplotlib']
missing_packages = [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]

if missing_packages:
    print('Instalando dependencias:', ', '.join(missing_packages))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
else:
    print('Dependencias ja disponiveis.')


# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path('wdbc.data')

base_features = [
    'radius',
    'texture',
    'perimeter',
    'area',
    'smoothness',
    'compactness',
    'concavity',
    'concave_points',
    'symmetry',
    'fractal_dimension',
]

columns = (
    ['id', 'diagnosis']
    + [f'{feature}_mean' for feature in base_features]
    + [f'{feature}_se' for feature in base_features]
    + [f'{feature}_worst' for feature in base_features]
)

df_raw = pd.read_csv(DATA_PATH, header=None, names=columns)
df = df_raw.copy()
df['diagnosis_binary'] = df['diagnosis'].map({'B': 0, 'M': 1}).astype(int)

feature_columns = [col for col in df.columns if col not in ['id', 'diagnosis', 'diagnosis_binary']]
audit_df = df[['id', 'diagnosis', 'diagnosis_binary'] + feature_columns]

print('Volumetria final dos dados tratados')
print(f'Linhas: {audit_df.shape[0]}')
print(f'Colunas totais: {audit_df.shape[1]}')
print(f'Variaveis morfologicas: {len(feature_columns)}')
print('Distribuicao do alvo codificado:')
print(audit_df['diagnosis_binary'].value_counts().rename(index={0: 'Benigno (0)', 1: 'Maligno (1)'}))
print(f'Valores ausentes: {int(audit_df.isna().sum().sum())}')

audit_df.head()


# ## Requisito II: Diagnostico de Assimetria Biologica
# 
# A variavel auditada e `area_mean`, que representa a area media do nucleo celular.
# 
# Para comprovar assimetria, serao calculadas metricas de tendencia central, dispersao e cauda extrema. Se a media ficar acima da mediana e os valores maximos ficarem muito distantes do centro, ha evidencia de cauda direita.

# In[ ]:


area = audit_df['area_mean']

area_stats = pd.Series({
    'media': area.mean(),
    'mediana': area.median(),
    'desvio_padrao': area.std(),
    'minimo': area.min(),
    'q1': area.quantile(0.25),
    'q3': area.quantile(0.75),
    'maximo': area.max(),
    'assimetria_skew': area.skew(),
    'media_menos_mediana': area.mean() - area.median(),
    'maximo_sobre_mediana': area.max() / area.median(),
    'coeficiente_variacao': area.std() / area.mean(),
})

print('Metricas da variavel area_mean')
display(area_stats.to_frame('valor'))

if area_stats['assimetria_skew'] > 0:
    print('Diagnostico: distribuicao com assimetria positiva, isto e, cauda direita.')
else:
    print('Diagnostico: nao ha evidencia de assimetria positiva pela metrica skew.')


# In[ ]:


def gaussian_kde_manual(values, grid_size=300):
    values = np.asarray(values, dtype=float)
    n = values.size
    std = values.std(ddof=1)
    bandwidth = 1.06 * std * (n ** (-1 / 5))
    bandwidth = bandwidth if bandwidth > 0 else 1.0
    grid = np.linspace(values.min(), values.max(), grid_size)
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z ** 2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))
    return grid, density

x_grid, density = gaussian_kde_manual(area)

plt.figure(figsize=(10, 5))
plt.plot(x_grid, density, color='#0f766e', linewidth=2.5, label='Densidade estimada')
plt.fill_between(x_grid, density, color='#99f6e4', alpha=0.45)
plt.axvline(area.mean(), color='#dc2626', linestyle='--', linewidth=2, label=f'Media = {area.mean():.2f}')
plt.axvline(area.median(), color='#2563eb', linestyle=':', linewidth=2.5, label=f'Mediana = {area.median():.2f}')
plt.title('Densidade da area media do nucleo celular (area_mean)')
plt.xlabel('area_mean')
plt.ylabel('densidade')
plt.legend()
plt.grid(alpha=0.25)
plt.show()


# ## Requisito III: Isolamento Robusto de Anomalias
# 
# Como `area_mean` apresenta assimetria biologica, nao sera usado z-score nem premissa de normalidade.
# 
# Metodo escolhido: **IQR**, abordagem nao-parametrica.
# 
# Limite superior de corte:
# 
# `Q3 + 1.5 * IQR`
# 
# Pacientes acima desse limite serao tratados como subgrupo clinicamente atipico para area celular.

# In[ ]:


q1 = area.quantile(0.25)
q3 = area.quantile(0.75)
iqr = q3 - q1
upper_cutoff = q3 + 1.5 * iqr

outliers_area = audit_df[audit_df['area_mean'] > upper_cutoff].copy()
malignant_prevalence = outliers_area['diagnosis_binary'].mean()

print('Isolamento robusto por IQR')
print(f'Q1: {q1:.4f}')
print(f'Q3: {q3:.4f}')
print(f'IQR: {iqr:.4f}')
print(f'Limite numerico de corte: {upper_cutoff:.4f}')
print(f'Pacientes atipicos isolados: {len(outliers_area)} de {len(audit_df)}')
print(f'Taxa de prevalencia de tumores malignos no subgrupo: {malignant_prevalence:.2%}')

outliers_area[['id', 'diagnosis', 'diagnosis_binary', 'area_mean', 'radius_mean', 'perimeter_mean']].sort_values('area_mean', ascending=False)


# ## Requisito IV: Mapeamento de Multicolinearidade
# 
# A multicolinearidade ocorre quando atributos carregam informacao muito semelhante. Em imagens celulares, medidas como raio, perimetro e area tendem a crescer juntas.
# 
# A matriz abaixo quantifica a correlacao cruzada entre todas as variaveis morfologicas. Valores proximos de `1` ou `-1` indicam forte redundancia informacional.

# In[ ]:


corr = audit_df[feature_columns].corr(method='pearson')

plt.figure(figsize=(14, 12))
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04, label='correlacao de Pearson')
plt.xticks(range(len(feature_columns)), feature_columns, rotation=90, fontsize=8)
plt.yticks(range(len(feature_columns)), feature_columns, fontsize=8)
plt.title('Mapa de correlacao entre variaveis morfologicas WDBC')
plt.tight_layout()
plt.show()

corr_pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .rename('correlacao')
    .reset_index()
    .rename(columns={'level_0': 'variavel_1', 'level_1': 'variavel_2'})
)

high_corr = corr_pairs[corr_pairs['correlacao'].abs() >= 0.90].sort_values('correlacao', key=lambda s: s.abs(), ascending=False)
print(f'Pares com |correlacao| >= 0.90: {len(high_corr)}')
display(high_corr.head(20))


# ## Parecer Tecnico e Etico
# 
# Com base na auditoria proposta:
# 
# 1. A base possui estrutura adequada para classificacao binaria apos codificacao do diagnostico (`M=1`, `B=0`).
# 2. A variavel `area_mean` tende a apresentar assimetria positiva, o que torna inadequado usar metodos parametricos simples para outliers.
# 3. O isolamento por IQR e mais robusto porque nao assume distribuicao normal. A prevalencia maligna dentro do subgrupo atipico deve ser interpretada como sinal clinico relevante, nao como decisao automatica.
# 4. A matriz de correlacao deve evidenciar redundancia entre medidas geometricas, especialmente entre raio, perimetro e area. Isso pode afetar modelos lineares e interpretabilidade se a base for usada futuramente em classificadores.
# 5. A base nao contem variaveis demograficas como idade, raca/cor, territorio, renda ou acesso previo a saude. Portanto, nao permite auditar diretamente justica algoritmica entre grupos populacionais.
# 
# **Parecer:** a base e viavel para estudo exploratorio e prototipagem controlada, mas seu uso em triagem na rede publica exige cautela. Antes de qualquer aplicacao operacional, seria necessario validar desempenho em dados locais, medir vieses por subgrupos demograficos, garantir supervisao clinica, documentar limites do sistema e evitar que o algoritmo substitua avaliacao medica.
