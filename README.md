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
# 1. Créer un environnement virtuel isolé
python -m venv env
source env/bin/activate        # macOS / Linux
# env\Scripts\activate         # Windows

# 2. Installer les dépendances exactes
pip install -r requirements.txt

# 3. Lancer Jupyter
jupyter notebook tp3_deep_learning.ipynb
```

> L'environnement virtuel est isolé — il n'utilise pas les packages installés sur le système.

## Dépendances (requirements.txt)

| Package | Version |
|---|---|
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| tensorflow | 2.21.0 |
| numpy | 2.5.0 |
| matplotlib | 3.11.0 |
| notebook | 7.6.0 |

Python ≥ 3.10 requis.

> **Note GPU** : torch et tensorflow détectent automatiquement CUDA (NVIDIA) et MPS (Apple Silicon). Sur CPU, le notebook fonctionne mais l'entraînement est plus lent.
