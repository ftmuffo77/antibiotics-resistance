import pandas as pd

# ======================================================
# 1. LER METADADOS
# ======================================================
meta_path = "/kaggle/input/gono-unitigs/metadata.csv"
meta = pd.read_csv(meta_path)

# Normalizar Sample_ID
meta["Sample_ID"] = (
    meta["Sample_ID"]
    .astype(str)
    .str.strip()
    .str.replace(r"[^A-Za-z0-9]", "", regex=True)
)

# ======================================================
# 2. TABELA DE CLUSTERS (que você gerou do PCA/KMeans)
# ======================================================
df_cluster = pd.DataFrame({
    "Sample_ID": df.index,   # df é o dataframe transposto usado no ML
    "cluster": clusters
})

# ======================================================
# 3. MERGE METADATA ↔ CLUSTERS
# ======================================================
merged = meta.merge(df_cluster, on="Sample_ID", how="inner")

print("Amostras relacionadas:", merged.shape)
print(merged.head())

# ======================================================
# 4. RESUMO POR CLUSTER
# ======================================================
# Frequência por país (se existir)
if "country" in merged.columns:
    print("\nDistribuição por país:")
    print(merged.groupby("cluster")["country"].value_counts())

# Frequência por ano
if "year" in merged.columns:
    print("\nDistribuição por ano:")
    print(merged.groupby("cluster")["year"].describe())
    
# Qualquer outra coluna disponível no metadata
for col in merged.columns:
    if col not in ["Sample_ID", "cluster"] and merged[col].dtype == object:
        print(f"\n{col} por cluster:")
        print(merged.groupby("cluster")[col].value_counts().head())
