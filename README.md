# TP3 — Deep Learning : TensorFlow vs PyTorch

Travaux pratiques comparant les deux frameworks deep learning majeurs sur des problèmes d'optimisation et de classification d'images.

## Contenu

| Fichier | Description |
|---|---|
| `tp3_deep_learning.ipynb` | Notebook Jupyter — 6 sections interactives |

## Sections couvertes

1. **Tenseurs** — création, opérations, device (CPU/GPU/MPS)
2. **Autograd** — différentiation automatique TF (`GradientTape`) vs PyTorch (`backward`)
3. **Optimisation** — descente de gradient sur la fonction de Rosenbrock
4. **Réseau de neurones** — MLP et CNN sur MNIST
5. **Boucle d'entraînement** — loop custom sans `model.fit`
6. **Comparaison** — graphes de convergence, temps d'exécution, syntaxe

## Points clés

- **TensorFlow** : `@tf.function`, `GradientTape`, graphe XLA statique
- **PyTorch** : `autograd`, graphe dynamique (eager), `.backward()`
- **Détection device** : CUDA → MPS (Apple Silicon) → CPU automatique

## Lancer le notebook

```bash
# Créer l'environnement
python -m venv env
source env/bin/activate

# Installer les dépendances
pip install torch torchvision tensorflow matplotlib jupyter

# Lancer Jupyter
jupyter notebook tp3_deep_learning.ipynb
```

## Dépendances

- Python ≥ 3.10
- torch ≥ 2.0
- torchvision
- tensorflow ≥ 2.12
- matplotlib
- numpy
