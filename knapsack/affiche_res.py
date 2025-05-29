import numpy as np
import matplotlib.pyplot as plt

with open("print_res.txt") as f:
    lines = f.readlines()
    
for line in lines:
    l = list(map(float, line.split(";")))
    
# Création du graphique
fig, ax1 = plt.subplots(figsize=(10, 6))

# Tracer la première courbe avec l'axe des ordonnées à gauche
color = 'tab:blue'
ax1.set_xlabel('X')
ax1.set_ylabel('Y1', color=color)
ax1.plot(range(len(l)-1), l[1:], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Ajouter une ligne horizontale
ax1.axhline(y=l[0], color='gray', linestyle='--', linewidth=0.8)  # Ligne horizontale à y=0

# Titre du graphique
plt.title('Graphique avec deux axes des ordonnées')

# Afficher le graphique
plt.show()
