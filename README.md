# Modèle génératif par diffusion pour séries temporelles financières via MBG

Reproduction et extension de **Kim, Choi, Kim (2025)** — *A diffusion-based generative model for financial time series via geometric Brownian motion* ([arXiv:2507.19003](https://arxiv.org/abs/2507.19003)).

> Projet réalisé dans le cadre du cours **Machine Learning for Finance, Theoretical Foundations** (ENSAE, enseignants : J.-D. Fermanian et H. Pham, année 2025/2026).

---

## Table des matières

1. [Contexte](#contexte)
2. [Structure du projet](#structure-du-projet)
3. [Installation](#installation)
4. [Utilisation rapide](#utilisation-rapide)
   - [Option A — CPU](#option-a--reproduction-cpu-recommandée-pour-tester-sans-gpu)
   - [Option B — GPU](#option-b--test-et-reproduction-sur-gpu)
   - [Option C — Pipeline manuel](#option-c--pipeline-manuel-pas-à-pas)
5. [Reproduction complète du papier](#reproduction-complète-du-papier)
6. [Dataset alternatif : CAC 40](#dataset-alternatif--cac-40-actions-françaises)
7. [Extensions](#extensions)
8. [Configuration](#configuration)
9. [Métriques d&#39;évaluation](#métriques-dévaluation)
10. [Pièges courants](#pièges-courants)
11. [Équipe](#équipe)
12. [Références](#références)

---

## Contexte

Le papier de Kim *et al.* (2025) propose un modèle génératif par diffusion dédié aux séries financières. Sa contribution principale consiste à injecter un bruit **multiplicatif de type Mouvement Brownien Géométrique (MBG)** dans le processus de diffusion avant (forward SDE), au lieu des bruits additifs classiques (VE / VP). Après annulation du drift (μ_t = σ_t² / 2), le modèle se ramène à une VE-SDE dans l'espace des **log-prix**, ce qui permet de capturer naturellement trois faits stylisés fondamentaux :

- **Queues lourdes** (exposant de puissance α ≈ 4,35 sur le S&P 500) ;
- **Clustering de volatilité** (décroissance lente en loi de puissance de l'autocorrélation de |r_t|) ;
- **Effet de levier** (corrélation négative persistante entre rendements et volatilité future).

Le réseau de score suit l'architecture **CSDI Transformer** (Tashiro *et al.*, 2021) : 128 canaux convolutifs, 256 dimensions d'embedding de diffusion, 64 dimensions d'embedding de caractéristiques, 4 blocs résiduels à portes.

Le présent dépôt implémente :

- le **pipeline de données** (téléchargement via `yfinance`, filtrage par ancienneté, fenêtres glissantes) ;
- les **trois SDE** (VE, VP, MBG) avec les trois schedules de σ (linéaire, exponentiel, cosinus) ;
- le **réseau de score** CSDI et la **perte DSM** (denoising score matching) ;
- l'**échantillonneur inverse** (Euler-Maruyama) ;
- le calcul des **trois faits stylisés** et la génération automatique des figures de comparaison.

---

## Structure du projet

```
Projet_ML_in_Finance/
├── README.md                   # Ce fichier
├── requirements.txt            # Dépendances Python
├── LICENSE
├── .gitignore
│
├── configs/                    # Fichiers de configuration YAML
│   ├── paper.yaml              # S&P 500 — papier, GPU, L=2048, 1000 epochs
│   ├── cpu_small.yaml          # S&P 500 — version CPU réduite ~10×
│   ├── paper_cac40.yaml        # CAC 40  — papier, GPU, L=2048, 1000 epochs
│   ├── cpu_cac40.yaml          # CAC 40  — version CPU réduite ~10×
│   └── default.yaml            # Configuration flexible (extensions)
│
├── src/                        # Code source du modèle
│   ├── data.py                 # Pipeline yfinance + fenêtres glissantes
│   ├── sde.py                  # VE, VP, MBG (+ CEV pour extension)
│   ├── model.py                # Réseau de score Transformer CSDI
│   ├── diffusion.py            # Perte DSM + échantillonneur inverse Euler
│   ├── stylized_facts.py       # α, autocorrélation |r|, L(k)
│   └── plotting.py             # Figures comparatives
│
├── scripts/                    # Points d'entrée en ligne de commande
│   ├── train.py                # Entraînement d'un modèle
│   ├── generate.py             # Génération d'échantillons synthétiques
│   ├── evaluate.py             # Calcul des faits stylisés et figures
│   ├── reproduce_all.sh        # S&P 500 — grille 3×3 complète (GPU)
│   ├── reproduce_cpu.sh        # S&P 500 — 3 configurations clés (CPU)
│   ├── reproduce_cac40_all.sh  # CAC 40  — grille 3×3 complète (GPU)
│   └── reproduce_cac40.sh      # CAC 40  — 3 configurations clés (CPU)
│
├── notebooks/                  # Notebooks d'analyse exploratoire
├── data/
│   ├── raw/                    # Données brutes yfinance (non versionnées)
│   └── processed/              # Fenêtres prétraitées (non versionnées)
├── experiments/
│   └── checkpoints/            # Poids entraînés (non versionnés)
├── results/
│   ├── figures/                # Figures produites par evaluate.py
│   └── *.npy                   # Échantillons générés (non versionnés)
└── report/                     # Rapport LaTeX et slides
```

Les fichiers volumineux (données, points de contrôle, échantillons, figures) sont exclus par `.gitignore`. Les dossiers vides sont conservés via des fichiers `.gitkeep`.

---

## Installation

### Prérequis

- Python ≥ 3.10
- (Optionnel) GPU CUDA pour la reproduction à l'échelle du papier

### Mise en place

```bash
# 1. Cloner le dépôt
git clone <URL_DU_DEPOT>
cd Projet_ML_in_Finance

# 2. Créer un environnement virtuel isolé
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

### Dépendances principales

| Paquet         | Rôle                                      |
| -------------- | ------------------------------------------ |
| `torch`      | Réseau de score + entraînement           |
| `yfinance`   | Téléchargement des prix historiques      |
| `pandas`     | Manipulation des séries temporelles       |
| `numpy`      | Calcul numérique                          |
| `scipy`      | Autocorrélation, statistiques             |
| `powerlaw`   | Estimation de l'exposant de queue α       |
| `matplotlib` | Figures                                    |
| `pyyaml`     | Lecture des configurations                 |
| `einops`     | Réarrangements tensoriels dans le modèle |
| `tqdm`       | Barres de progression                      |

---

## Utilisation rapide

### Option A — Reproduction CPU (recommandée pour tester sans GPU)

Trois configurations clés à échelle réduite (≈ 2 h 30 au total sur un i5-1145G7) :

```bash
bash scripts/reproduce_cpu.sh
```

Cela exécute dans l'ordre :

1. Téléchargement de 20 grandes capitalisations anciennes (≥ 30 ans d'historique) ;
2. **Pour chacun** des trois modèles (**MBG + cosinus**, **MBG + exponentiel**, **VE + cosinus**), de bout en bout dans la même itération :
   - Entraînement (checkpoint sauvegardé dès la meilleure loss) ;
   - Génération de 120 séries synthétiques ;
   - Évaluation des faits stylisés + figure propre au modèle dans `results/figures/cpu_<sde>_<schedule>.png`.
3. Figure agrégée comparant les trois modèles : `results/figures/cpu_reproduction.png`.

Dès qu'un modèle termine (~50 min à 2 h selon la config), sa figure est disponible — plus besoin d'attendre la fin des trois runs pour voir un résultat. Le script est **resumable** : checkpoints, samples et figures déjà présents sont automatiquement skippés, donc une relance après interruption reprend là où elle s'est arrêtée.

Pour un test rapide (quelques minutes) :

```bash
EPOCHS=50 bash scripts/reproduce_cpu.sh
```

### Option B — Test et reproduction sur GPU

Le papier a été entraîné sur GPU (L=2048, 1000 epochs). Si vous disposez d'une carte CUDA, procédez par paliers croissants pour valider l'installation avant de lancer la grille complète.

#### B.1 Vérifier que CUDA est disponible

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 2), 'GB') if torch.cuda.is_available() else None"
```

Sortie attendue (exemple) :

```
CUDA: True | device: NVIDIA GeForce RTX 3090
VRAM: 24.0 GB
```

Si `CUDA: False`, réinstaller `torch` avec la build CUDA adaptée à votre driver (voir [pytorch.org/get-started](https://pytorch.org/get-started/locally/)).

> Dans `configs/paper.yaml`, le champ `train.device: cuda` est déjà fixé. Les scripts `train.py` et `generate.py` détectent automatiquement le GPU et retombent sur CPU si CUDA n'est pas disponible.

#### B.2 Smoke-test (≈ 2 min) — 1 modèle, 5 epochs

Vérifie que le forward/backward passe sans OOM et qu'une figure est produite :

```bash
# 1. Préparer les données (cache persistant)
python -m src.data --universe sp500 --min-years 40 --seq-len 2048 --stride 400

# 2. Entraînement éclair (5 epochs, ~2 min sur RTX 3090)
python scripts/train.py --config configs/paper.yaml \
    --sde gbm --schedule cosine --epochs 5 --tag smoke_gpu

# 3. Génération rapide (8 échantillons au lieu de 120)
python scripts/generate.py --config configs/paper.yaml \
    --ckpt experiments/checkpoints/smoke_gpu.pt \
    --n-samples 8 --out results/samples_smoke.npy

# 4. Évaluation
python scripts/evaluate.py \
    --real data/processed/sp500_L2048_S400.npz \
    --runs smoke=results/samples_smoke.npy \
    --out results/figures/smoke_gpu.png
```

Si `results/figures/smoke_gpu.png` est généré sans erreur, le pipeline GPU est fonctionnel. Les faits stylisés seront mauvais à 5 epochs — c'est normal, le but est uniquement de valider l'exécution de bout en bout.

#### B.3 Test d'une configuration cible (≈ 30 min — 2 h selon GPU)

Une seule configuration à l'échelle du papier (MBG + cosinus, la meilleure selon §4.2) :

```bash
EPOCHS=300 python scripts/train.py --config configs/paper.yaml \
    --sde gbm --schedule cosine --tag sp500_gbm_cosine

python scripts/generate.py --config configs/paper.yaml \
    --ckpt experiments/checkpoints/sp500_gbm_cosine.pt \
    --n-samples 120 --out results/samples_gbm_cosine.npy

python scripts/evaluate.py \
    --real data/processed/sp500_L2048_S400.npz \
    --runs gbm_cosine=results/samples_gbm_cosine.npy \
    --out results/figures/gbm_cosine_gpu.png
```

Attendu : α ∈ [3,0 ; 5,0] (la cible papier est 3,78 ; l'empirique S&P 500 est 4,35).

#### B.4 Reproduction complète sur GPU (grille 3×3)

Voir la section [Reproduction complète du papier](#reproduction-complète-du-papier) ci-dessous. Temps indicatifs :

| GPU              | VRAM | 1000 epochs × 9 modèles |
| ---------------- | ---- | ------------------------- |
| RTX 3060 (12 GB) | 12   | ~4 jours                  |
| RTX 3090 / 4090  | 24   | ~1,5 à 2 jours           |
| A100 40 GB       | 40   | ~20 h                     |

#### B.5 Surveillance et dépannage GPU

```bash
# Dans un autre terminal pendant l'entraînement
watch -n 2 nvidia-smi
```

**OOM (out-of-memory)** — réduire `train.batch_size` de 64 → 32 dans `configs/paper.yaml`, ou passer à `configs/cpu_small.yaml` (L=256) avec `device: cuda` pour un modèle plus léger tout en utilisant le GPU.

**GPU sous-utilisé (< 50 %)** — le dataloader CPU est souvent le goulot ; augmenter `train.num_workers` si exposé, sinon ignorer (l'entraînement est déjà rapide à L=2048).

**Forcer le CPU même avec un GPU présent** — éditer `configs/paper.yaml` et remplacer `device: cuda` par `device: cpu`.

### Option C — Pipeline manuel pas à pas

```bash
# 1. Préparer le dataset
python -m src.data --universe sp500 --min-years 40 --seq-len 2048 --stride 400

# 2. Entraîner un modèle
python scripts/train.py --config configs/paper.yaml --sde gbm --schedule cosine

# 3. Générer 120 échantillons
python scripts/generate.py \
    --config configs/paper.yaml \
    --ckpt experiments/checkpoints/sp500_gbm_cosine.pt \
    --n-samples 120 \
    --out results/samples_gbm_cosine.npy

# 4. Évaluer et tracer les figures
python scripts/evaluate.py \
    --real data/processed/sp500_L2048_S400.npz \
    --runs gbm_cosine=results/samples_gbm_cosine.npy \
    --out results/figures/gbm_cosine.png
```

---

## Reproduction complète du papier

Pour la grille 3×3 complète (VE / VP / MBG × linéaire / exponentiel / cosinus) à l'échelle du papier :

```bash
bash scripts/reproduce_all.sh
```

Le script traite les 9 configurations **une par une, de bout en bout** : pour chaque (SDE, schedule) il enchaîne `train` → `generate` → `evaluate` et produit une figure dédiée `results/figures/sp500_<sde>_<schedule>.png` avant de passer à la suivante. Une fois les 9 modèles terminés, la figure agrégée 3×3 est écrite dans `results/figures/sp500_grid_3x3.png`.

Avantages :

- **Résultats intermédiaires** dès qu'un modèle termine (~1,5 à 2 h sur RTX 3090) — pas besoin d'attendre la fin des 9 runs pour voir une figure.
- **Resumable** : si la session est interrompue (timeout, OOM, reboot), relancer la même commande reprend là où elle s'est arrêtée (checkpoints, samples et figures déjà présents sont skippés).

> **Attention** : 9 modèles × 1000 epochs × L=2048 représentent **plusieurs jours-GPU**. Pour une vérification rapide de l'implémentation, utiliser :
>
> ```bash
> EPOCHS=50 bash scripts/reproduce_all.sh
> ```

### Résultats attendus (papier, §4.2 / Figure 5)

Exposant de queue α estimé par `powerlaw.Fit` sur les 5 % de |r_t| les plus grands.

Benchmark S&P 500 réel : **α ≈ 4,35**.

| SDE \ Schedule   | Linéaire | Exponentiel       | Cosinus           |
| ---------------- | --------- | ----------------- | ----------------- |
| **VE**     | 8,96      | 8,49              | 4,14              |
| **MBG** ⭐ | 3,06      | **4,62** ✨ | **3,78** ✨ |

Les configurations **MBG + exponentiel** et **MBG + cosinus** se rapprochent le plus de l'empirique (α ≈ 4,35). Une tolérance de ± 20 % est acceptable en raison des différences de seed et de période de téléchargement.

---

## Dataset alternatif : CAC 40 (actions françaises)

En plus de la reproduction S&P 500, le dépôt propose un **second flux indépendant** sur le **CAC 40**, l'indice phare d'Euronext Paris. Les deux flux utilisent le même code source (`src/`) et le même pipeline ; seul le bloc `data.universe` distingue leurs artefacts.

### Vue d'ensemble des flux parallèles

| Univers                | Config CPU       | Config GPU (papier)    | Script CPU                | Script GPU (3×3)             |
| ---------------------- | ---------------- | ---------------------- | ------------------------- | ---------------------------- |
| **S&P 500** (original) | `cpu_small.yaml` | `paper.yaml`           | `reproduce_cpu.sh`        | `reproduce_all.sh`           |
| **CAC 40** (ajouté)    | `cpu_cac40.yaml` | `paper_cac40.yaml`     | `reproduce_cac40.sh`      | `reproduce_cac40_all.sh`     |

Les deux flux peuvent s'exécuter côte à côte : **checkpoints, samples, fenêtres prétraitées et figures sont namespacés** par univers (`cac40_*` vs. `sp500_*`) et ne se recouvrent jamais.

### Spécificités du CAC 40

- **40 constituants** (vs. ~500 pour le S&P 500) récupérés automatiquement via [`fetch_cac40_tickers`](src/data.py) (Wikipedia), avec fallback sur 20 blue-chips à long historique (AI.PA, BN.PA, MC.PA, OR.PA, SAN.PA, TTE.PA, …).
- **Historique Yahoo** : la plupart des tickers `.PA` remontent à 1987–1988 → `min_years` est abaissé de **40 (papier) à 25** dans `cpu_cac40.yaml` et `paper_cac40.yaml`. Le reste des hyperparamètres est strictement identique au papier.
- **Pas de benchmark α dans le papier** : le rapport comparera la cohérence interne (MBG vs. VE, effet du schedule) et la proximité vis-à-vis du CAC 40 **réel**, sans référence paper-α.

### Flux CPU CAC 40 (≈ 2 h 30 sur i5)

Équivalent de l'Option A mais sur le CAC 40. Exécute trois configurations (**MBG + cosinus**, **MBG + exponentiel**, **VE + cosinus**) de bout en bout sur 20 blue-chips français hard-codés dans `cpu_cac40.yaml` :

```bash
bash scripts/reproduce_cac40.sh
# Smoke test rapide :
EPOCHS=50 bash scripts/reproduce_cac40.sh
```

Artefacts produits :

- `experiments/checkpoints/cac40_<sde>_<sched>.pt` — 3 checkpoints
- `results/samples_cac40_<sde>_<sched>.npy` — 3 × 120 séries synthétiques
- `results/figures/cac40_<sde>_<sched>.png` — une figure par modèle
- `results/figures/cac40_reproduction.png` — figure agrégée comparative

### Flux GPU paper-scale CAC 40 (grille 3×3 complète)

Équivalent de l'Option B.4 / `reproduce_all.sh` mais sur le CAC 40. Exécute la grille complète **VE / VP / MBG × linéaire / exponentiel / cosinus** à L=2048 et 1000 epochs :

```bash
bash scripts/reproduce_cac40_all.sh
# Smoke test rapide :
EPOCHS=50 bash scripts/reproduce_cac40_all.sh
```

Le script est **resumable** (checkpoints, samples et figures déjà présents sont skippés) et produit des résultats intermédiaires après chaque modèle.

Artefacts produits :

- `data/processed/cac40_L2048_S400.npz` — fenêtres CAC 40
- `experiments/checkpoints/cac40_<sde>_<sched>.pt` — 9 checkpoints
- `results/samples_cac40_<sde>_<sched>.npy` — 9 × 120 séries synthétiques
- `results/figures/cac40_<sde>_<sched>.png` — 9 figures par modèle
- `results/figures/cac40_grid_3x3.png` — figure agrégée 3×3

> **Attention** : comme pour le S&P 500, 9 modèles × 1000 epochs × L=2048 représentent **plusieurs jours-GPU**. En pratique le dataset CAC 40 est plus petit (moins de tickers → moins de fenêtres), donc l'entraînement est proportionnellement plus rapide mais le risque d'under-fitting est plus élevé.

### Coexistence des deux flux

| Artefact           | S&P 500                                  | CAC 40                                        |
| ------------------ | ---------------------------------------- | --------------------------------------------- |
| Données brutes     | `data/raw/sp500.parquet`                 | `data/raw/cac40.parquet`                      |
| Fenêtres           | `data/processed/sp500_L*.npz`            | `data/processed/cac40_L*.npz`                 |
| Checkpoints        | `experiments/checkpoints/sp500_*.pt`     | `experiments/checkpoints/cac40_*.pt`          |
| Samples            | `results/samples_{sde}_{sched}.npy`      | `results/samples_cac40_{sde}_{sched}.npy`     |
| Figures par modèle | `results/figures/sp500_*.png`            | `results/figures/cac40_*.png`                 |
| Grille 3×3        | `results/figures/sp500_grid_3x3.png`     | `results/figures/cac40_grid_3x3.png`          |

Vous pouvez exécuter par exemple `bash scripts/reproduce_cpu.sh` puis `bash scripts/reproduce_cac40.sh` sur la même machine, dans n'importe quel ordre — les caches et figures des deux univers sont isolés.

---

## Extensions

Conformément aux consignes de validation du cours (*changer le jeu de données OU un choix de modélisation clé*), deux extensions sont implémentées :

### A. Changement de jeu de données : crypto-actifs

Les crypto-monnaies présentent des queues encore plus lourdes que les actions.

```bash
python -m src.data --universe crypto --tickers BTC-USD,ETH-USD \
    --min-years 0 --seq-len 2048 --stride 400

python scripts/train.py --config configs/default.yaml \
    --sde gbm --schedule cosine --universe crypto
```

### B. Changement de modélisation : SDE CEV

Le modèle **Constant Elasticity of Variance** généralise le MBG via un paramètre γ ∈ {0,5 ; 0,7 ; 1,0} (γ = 1 correspond au MBG).

Dans `configs/default.yaml`, fixer :

```yaml
sde:
  type: cev
  cev_gamma: 0.7
```

Puis :

```bash
python scripts/train.py --config configs/default.yaml --sde cev
```

---

## Configuration

Cinq fichiers YAML sont fournis dans `configs/` :

| Fichier              | Univers    | Usage                                                                              |
| -------------------- | ---------- | ---------------------------------------------------------------------------------- |
| `paper.yaml`         | S&P 500    | Hyperparamètres verrouillés sur le papier (§4). **Ne pas modifier.**              |
| `cpu_small.yaml`     | S&P 500    | Version CPU réduite ~10× (L=256, 2 blocs, 300 epochs)                             |
| `paper_cac40.yaml`   | CAC 40     | Miroir paper-scale adapté au CAC 40 (`min_years=25`)                              |
| `cpu_cac40.yaml`     | CAC 40     | Version CPU réduite ~10× sur 20 blue-chips français                               |
| `default.yaml`       | (flexible) | Configuration pour les extensions                                                  |

### Hyperparamètres du papier (§4)

| Paramètre                         | Valeur          |
| ---------------------------------- | --------------- |
| Longueur de séquence L            | 2048            |
| Pas de la fenêtre glissante       | 400             |
| σ_min, σ_max                     | 0,01, 1,0       |
| Horizon de diffusion avant T       | 1,0             |
| Pas inverses N                     | 2000            |
| Taille de batch                    | 64              |
| Epochs                             | 1000            |
| Canaux convolutifs                 | 128             |
| Dim. embedding de diffusion        | 256             |
| Dim. embedding de caractéristique | 64              |
| Blocs résiduels                   | 4               |
| Optimiseur                         | Adam, lr = 1e-4 |

---

## Métriques d'évaluation

Trois faits stylisés sont calculés par `src/stylized_facts.py` :

1. **Queues lourdes** — exposant α estimé sur les 5 % supérieurs de |r_t| via `powerlaw.Fit`. Les estimateurs Hill et `powerlaw` peuvent différer de ± 0,5 ; le rapport inclut les deux.
2. **Clustering de volatilité** — autocorrélation de |r_t| pour les décalages 1…1000, tracée en échelle log-log (décroissance en loi de puissance attendue).
3. **Effet de levier** —

   $$
   L(k) = \frac{\mathbb{E}[r_t\, r_{t+k}^2 - r_t\, |r_t|^2]}{\mathbb{E}[|r_t|^2]^2}
   $$

   pour k = 0…100 (valeurs négatives persistantes attendues).

Le script `scripts/evaluate.py` produit un tableau comparatif α (nôtre vs papier) et une figure multi-panneaux superposant les courbes réelles et synthétiques.

---

## Pièges courants

1. **Biais de survie** : les constituants listés sur Wikipedia sont ceux présents *aujourd'hui*. Le papier hérite de ce biais ; nous le conservons pour la Phase 1.
2. **`auto_adjust` de yfinance** : doit rester à `True` pour des prix ajustés des dividendes et splits.
3. **Estimateurs de α** : différences de ± 0,5 entre `powerlaw.Fit` et Hill sont normales.
4. **Mémoire GPU** : L=2048, batch=64, 128 canaux ≈ 4 GB VRAM. Réduire le batch à 32 en cas d'OOM.
5. **Temps d'échantillonnage** : N=2000 × 120 échantillons × L=2048 → 10 à 30 min par modèle selon le GPU.
6. **Pollution du cache `data/raw/<universe>.parquet`** : le nom du fichier cache dépend uniquement de `universe`, pas de la liste de tickers. Si vous lancez d'abord `reproduce_cpu.sh` (qui hard-code 20 tickers) puis `reproduce_all.sh` (liste S&P 500 complète), le second **réutilise le cache à 20 tickers** et entraîne donc sur un dataset 20× plus petit que le papier (symptôme observé : `sp500: (420, 2048) windows` et α synthétiques 1,5 à 3,5 trop hauts). Même piège pour le CAC 40. **Solution** avant de basculer d'un sous-ensemble hard-codé à une liste complète :
   ```bash
   rm -f data/raw/<universe>.parquet data/processed/<universe>_L*.npz
   ```

---

## Équipe

Projet réalisé en binôme/trinôme dans le cadre du Master MIDS (Université Paris Cité) et du cursus ENSAE.

- Khac-Vinh **TANG** — *pipeline de données + extension crypto*
- Linh-Chi **VU** — *modèle CSDI + entraînement*
- Boutheina

---

## Références

- **Papier reproduit** : G. Kim, S.-Y. Choi, Y. Kim (2025). *A diffusion-based generative model for financial time series via geometric Brownian motion*. [arXiv:2507.19003](https://arxiv.org/abs/2507.19003).
- **CSDI** : Y. Tashiro *et al.* (2021). *Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation*. [arXiv:2107.03502](https://arxiv.org/abs/2107.03502).
- **Score-based SDE** : Y. Song *et al.* (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456).
- **DDPM** : J. Ho, A. Jain, P. Abbeel (2020). *Denoising Diffusion Probabilistic Models*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
- **GAN financier (baseline)** : S. Takahashi, Y. Chen, K. Tanaka-Ishii (2019). *Modeling financial time-series with generative adversarial networks*. *Physica A*, 527.

---

## Licence

Ce projet est distribué sous licence MIT — voir [LICENSE](LICENSE).

Le code est mis à disposition à des fins pédagogiques et de recherche. Les droits sur le papier reproduit appartiennent à leurs auteurs originaux.
======================================================================================================================================================
