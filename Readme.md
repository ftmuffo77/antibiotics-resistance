
Esse arqiuvo contém

* Introdução
* Estrutura dos dados
* Pipeline utilizada
* Resultados fatuais
* Reprodutibilidade
* Referências internas


---

# 🧬 Análise Genômica de *Neisseria gonorrhoeae* via Unitigs e Clusterização

## 📌 Sobre este projeto

Este repositório documenta o processamento, análise e clusterização de dados genômicos de *Neisseria gonorrhoeae* a partir de arquivos `.Rtab` contendo matrizes esparsas de **unitigs** (padrões de sequência).
O objetivo é identificar agrupamentos genômicos distintos e relacioná-los **exclusivamente** às variáveis presentes no arquivo real `metadata.csv`.

Nenhuma inferência biológica, epidemiológica ou funcional foi realizada além do que está explícito nos dados.

---

## 📁 Estrutura dos Dados

### **1. Arquivos `.Rtab`**

Contêm matrizes do tipo:

```
unitigs (linhas) × amostras (colunas)
```

Exemplo do formato original:

* 8873 unitigs
* 3971 amostras
* valores binários (0/1)

### **2. Arquivo `metadata.csv`**

Inclui as seguintes colunas (confirmadas no dataset):

```
Sample_ID
Year
Country
Continent
Beta.lactamase
Azithromycin
Ciprofloxacin
Ceftriaxone
Cefixime
Tetracycline
Penicillin
NG_MAST
Group
azm_mic
cip_mic
cro_mic
cfx_mic
tet_mic
pen_mic
log2_azm_mic
log2_cip_mic
log2_cro_mic
log2_cfx_mic
log2_tet_mic
log2_pen_mic
azm_sr
cip_sr
cro_sr
cfx_sr
tet_sr
pen_sr
```

Essas variáveis são usadas para cruzamento factual com os clusters.

---

## ⚙️ Pipeline Utilizada

### **1. Carregamento dos Dados**

```python
df = pd.read_csv("*.Rtab", sep=r"\s+", engine="python", index_col=0)
```

### **2. Transformação**

* Transposição para o formato **amostras × unitigs**
* Remoção de unitigs sem variabilidade

### **3. Redução de Dimensionalidade**

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(df)
```

### **4. Clusterização**

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)
```

### **5. Importância dos unitigs**

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300)
rf.fit(df, clusters)

top20 = pd.Series(rf.feature_importances_, index=df.columns)              \
          .sort_values(ascending=False).head(20)
```

### **6. Associação com Metadados**

```python
merged = metadata.merge(
    pd.DataFrame({"Sample_ID": df.index, "cluster": clusters}),
    on="Sample_ID", how="inner"
)
```

---

## 📊 Resultados (usando apenas o primeiro arquivo de amostra RTAB)

### **Clusterização**

Dois clusters foram identificados:

* **Cluster 0:** 39 amostras
* **Cluster 1:** 868 amostras

### **Distribuição Temporal (coluna `Year`)**

| Cluster | Count | Mean Year   | Median | Min  | Max  |
| ------- | ----- | ----------- | ------ | ---- | ---- |
| **0**   | 39    | **2001.49** | 1999   | 1997 | 2015 |
| **1**   | 868   | **2012.69** | 2015   | 1989 | 2017 |

> Esses valores são extraídos diretamente da coluna `Year` do metadado, sem inferências adicionais.

### **Unitigs mais relevantes**

Os 20 unitigs com maior contribuição para a separação entre clusters foram extraídos pelo Random Forest.
Essas sequências representam apenas padrões distintos observados no dataset — sem interpretação funcional atribuída neste repositório.

---

## 🔁 Reprodutibilidade

### **Instalação das dependências**

```bash
pip install pandas numpy scikit-learn
```

### **Execução completa**

Um script reprodutível está disponível em `analysis.ipynb` (ou adicione o notebook ao repo).

---

## 📌 Observações Importantes

* Nenhuma relação funcional, fenotípica ou epidemiológica foi inferida além das informações presentes no dataset.
* As análises são puramente descritivas e exploratórias.
* Todas as conclusões numéricas são derivadas exclusivamente do conteúdo real dos arquivos `.Rtab` e `metadata.csv`.

---

## 📎 Licença

Este projeto pode ser utilizado para fins de estudo, pesquisa e exploração analítica.

---


