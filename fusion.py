import pandas as pd

# Charger les deux fichiers
df1 = pd.read_csv("results.csv")
df2 = pd.read_csv("results_merged.csv")
df3 = pd.read_csv("results.csv")

# Concaténer et supprimer les doublons (optionnel)
merged = pd.concat([df1, df2], ignore_index=True).drop_duplicates()

# Sauver le résultat combiné
merged.to_csv("results_merged2.csv", index=False)