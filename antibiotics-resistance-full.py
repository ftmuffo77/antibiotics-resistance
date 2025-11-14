# ============================================================
# ANÁLISE COMPLETA DE UNITIGS + METADADOS
# PCA → KMeans → RF → MÉTRICAS SUPERVISIONADAS → MERGE METADATA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score
)

# ============================================================
# 1. CARREGAR DADOS .RTAB
# ============================================================

print("=== Lendo arquivo RTAB ===")

rtab_path = "/kaggle/input/gono-unitigs/cip_sr_gwas_filtered_unitigs.Rtab"

df = pd.read_csv(rtab_path, sep=r"\s+", engine="python", index_col=0)
print("RTAB original (unitigs x amostras):", df.shape)

# Transformar para (amostras × unitigs)
df = df.T
print("Transformado (amostras x unitigs):", df.shape)

# Remover unitigs sem variabilidade
df = df.loc[:, df.nunique() > 1]
print("Após remover unitigs constantes:", df.shape)

# ============================================================
# 2. PCA
# ============================================================
print("\n=== Executando PCA ===")

pca = PCA(n_components=10)
X_pca = pca.fit_transform(df)
print("Formato PCA:", X_pca.shape)

# ============================================================
# 3. K-MEANS (clusterização não supervisionada)
# ============================================================
print("\n=== KMeans ===")

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)
print("Clusters identificados:", np.unique(clusters))

# ============================================================
# 4. RANDOM FOREST – IMPORTÂNCIA DOS UNITIGS
# ============================================================

print("\n=== Importância dos unitigs (RF) ===")

rf_full = RandomForestClassifier(n_estimators=300, random_state=42)
rf_full.fit(df, clusters)

importance = pd.Series(rf_full.feature_importances_, index=df.columns)
top20 = importance.sort_values(ascending=False).head(20)

print("\nTOP 20 unitigs mais relevantes:")
print(top20)

# ============================================================
# 5. MÉTRICAS SUPERVISIONADAS (RF tentando prever clusters)
# ============================================================

print("\n=== Avaliação do Modelo Supervisionado ===")

X_train, X_test, y_train, y_test = train_test_split(
    df, clusters, test_size=0.25, random_state=42, stratify=clusters
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ---------- métricas completas ----------
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_proba)
bacc = balanced_accuracy_score(y_test, y_pred)
mcc  = matthews_corrcoef(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

pr_auc = average_precision_score(y_test, y_proba)

print("\n===== MÉTRICAS COMPLETAS =====")
print(f"Acurácia:              {acc:.4f}")
print(f"Precisão:              {prec:.4f}")
print(f"Recall:                {rec:.4f}")
print(f"F1-score:              {f1:.4f}")
print(f"Especificidade:        {specificity:.4f}")
print(f"Balanced Accuracy:     {bacc:.4f}")
print(f"MCC:                   {mcc:.4f}")
print(f"ROC-AUC:               {auc:.4f}")
print(f"PR-AUC:                {pr_auc:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# 6. VISUALIZAÇÕES
# ============================================================

# ---------- Matriz de confusão ----------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()

# ---------- Curva ROC ----------
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], "k--")
plt.title("Curva ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# ---------- Curva Precision-Recall ----------
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
plt.title("Curva Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 7. MERGE COM METADADOS
# ============================================================

print("\n=== Merge com Metadata ===")

meta = pd.read_csv("/kaggle/input/gono-unitigs/metadata.csv")

meta["Sample_ID"] = (
    meta["Sample_ID"]
    .astype(str)
    .str.strip()
    .str.replace(r"[^A-Za-z0-9]", "", regex=True)
)

df_cluster = pd.DataFrame({
    "Sample_ID": df.index,
    "cluster": clusters
})

merged = meta.merge(df_cluster, on="Sample_ID", how="inner")

print("Amostras relacionadas:", merged.shape)
print(merged.head())

# ============================================================
# 8. RESUMO FÁTICO POR CLUSTER
# ============================================================

# ----- Ano -----
if "Year" in merged.columns:
    print("\n=== Estatísticas por Ano ===")
    print(merged.groupby("cluster")["Year"].describe())

# ----- País -----
if "Country" in merged.columns:
    print("\n=== País por cluster (top 15) ===")
    print(merged.groupby("cluster")["Country"].value_counts().head(15))

# ----- Continente -----
if "Continent" in merged.columns:
    print("\n=== Continente por cluster ===")
    print(merged.groupby("cluster")["Continent"].value_counts())

# ----- MICs -----
mic_cols = [c for c in merged.columns if c.endswith("_mic") or c.startswith("log2_")]
if mic_cols:
    print("\n=== Médias de MIC por Cluster ===")
    print(merged.groupby("cluster")[mic_cols].mean())

# ----- S/R -----
sr_cols = [c for c in merged.columns if c.endswith("_sr")]
for col in sr_cols:
    print(f"\n=== {col} por cluster ===")
    print(merged.groupby("cluster")[col].value_counts(dropna=False))

print("\n=== Fim do Script ===")
