import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# ====================================================
# 1. LER CORRETAMENTE O ARQUIVO
# ====================================================
path = "/kaggle/input/gono-unitigs/cip_sr_gwas_filtered_unitigs.Rtab"

df = pd.read_csv(path, sep=r"\s+", engine="python", index_col=0)
print("Formato original (unitigs x amostras):", df.shape)

# ====================================================
# 2. TRANSFORMAR PARA (amostras x unitigs)
# ====================================================
df = df.T
print("Formato convertido (amostras x unitigs):", df.shape)

# ====================================================
# 3. REMOVER COLUNAS CONSTANTES
# ====================================================
df = df.loc[:, df.nunique() > 1]
print("Qtde de unitigs após remover constantes:", df.shape[1])

# ====================================================
# 4. PCA (redução dimensional)
# ====================================================
pca = PCA(n_components=10)
X_pca = pca.fit_transform(df)
print("PCA gerado:", X_pca.shape)

# ====================================================
# 5. CLUSTERIZAÇÃO K-Means
# ====================================================
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

print("Clusters encontrados:", np.unique(clusters))

# ====================================================
# 6. IMPORTÂNCIA DOS UNITIGS (RF)
# ====================================================
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(df, clusters)

importance = pd.Series(rf.feature_importances_, index=df.columns)
top20 = importance.sort_values(ascending=False).head(20)

print("\nTOP 20 unitigs mais relevantes:")
print(top20)
