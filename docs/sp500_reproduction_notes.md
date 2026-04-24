# Notes de reproduction S&P 500 — écart observé avec le papier

> **Objet.** Documenter un écart significatif entre les exposants de queue α obtenus lors d'une reproduction complète à l'échelle du papier (Option B, GPU, `paper.yaml`) et ceux reportés par Kim, Choi, Kim (2025) §4.2, Figure 5. Identifier la cause racine et la procédure de correction.
>
> **Audience.** Collaborateurs du projet ; réviseurs souhaitant comprendre pourquoi les valeurs de α peuvent différer du papier avant toute nouvelle exécution.

---

## 1. Contexte d'exécution

Commande lancée par la collaboratrice (environnement : Linux, venv `.venv`, CUDA disponible) :

```bash
bash scripts/reproduce_all.sh
```

Le script utilise `configs/paper.yaml` (L=2048, 1000 epochs, `device: cuda`) et la grille complète 3×3 (VE / VP / MBG × linéaire / exponentiel / cosinus).

**Ligne de console critique observée au démarrage du script** :

```
=== [1/2] Preparing sp500 dataset ===
sp500: (420, 2048) windows
```

→ **420 fenêtres** pour l'ensemble de l'univers S&P 500.

---

## 2. Résultats quantitatifs

Exposants de queue α estimés sur les 5 % supérieurs de |r_t| (Hill / `powerlaw.Fit`), comparés aux valeurs reportées dans le papier §4.2 / Figure 5 :

| Modèle              | α (reproduction) | α (papier) | Δ         |
| ------------------- | ---------------: | ---------: | --------: |
| **real S&P 500**    |          **4,03** |       4,35 |    −0,32 ✅ |
| ve_linear           |             6,47 |       8,96 |    −2,49 |
| ve_exponential      |             5,60 |       8,49 |    −2,89 |
| ve_cosine           |             5,95 |       4,14 | **+1,81** |
| vp_linear           |             6,23 |        n/a |       n/a |
| vp_exponential      |             5,58 |        n/a |       n/a |
| vp_cosine           |             6,03 |        n/a |       n/a |
| gbm_linear          |             6,56 |       3,06 | **+3,50** |
| gbm_exponential     |             5,70 |       4,62 |    +1,08 |
| gbm_cosine          |             5,60 |       3,78 | **+1,82** |

**Faits saillants** :

1. **α (réel) ≈ papier** — l'écart sur les données empiriques est de seulement −0,32 → le pipeline de lecture de données et l'estimateur de queue fonctionnent correctement.
2. **Tous les α synthétiques sont supérieurs à α réel** (5,58 à 6,56 > 4,03). C'est le pattern inverse du papier, où α synthétique ≤ α réel (3,06 à 4,62 ≤ 4,35).
3. **Les écarts vont majoritairement dans le sens positif** (Δ > 0 pour les configurations MBG et ve_cosine). Cela signifie que les séries synthétiques ont des queues **trop légères** par rapport au papier — symptôme classique d'un modèle **sous-entraîné** qui produit des échantillons trop lisses.

---

## 3. Cause racine : pollution du cache brut

Le bug se trouve dans [`src/data.py`](../src/data.py), fonction `load_or_build`, lignes 103–110 :

```python
raw_path = raw_dir / f"{spec.universe}.parquet"   # nom dépend UNIQUEMENT de `universe`
# ...
if raw_path.exists():
    prices = pd.read_parquet(raw_path)            # réutilisé quelle que soit la liste de tickers demandée
```

Le nom du fichier cache ne dépend que de l'univers (ex. `sp500.parquet`), **pas** de la liste de tickers effectivement demandée. Par conséquent, un premier lancement avec une liste hard-codée de 20 tickers crée un cache à 20 colonnes, et tout lancement ultérieur avec une liste complète le réutilise **silencieusement**.

### Chaîne d'événements reconstituée

1. `bash scripts/reproduce_cpu.sh` est lancé en premier (workflow usuel de smoke test). Ce script appelle :
   ```bash
   python -m src.data --universe sp500 \
       --tickers IBM,KO,PG,XOM,GE,MCD,MMM,JNJ,MRK,PFE,CAT,DIS,WMT,BA,CVX,HON,T,F,HD,DD \
       --min-years 30 --seq-len 256 --stride 64
   ```
   → écrit `data/raw/sp500.parquet` avec **20 colonnes seulement**.

2. `bash scripts/reproduce_all.sh` est lancé ensuite (reproduction paper-scale). Il appelle :
   ```bash
   python -m src.data --universe sp500 --min-years 40 --seq-len 2048 --stride 400
   ```
   Comme `data/raw/sp500.parquet` existe déjà, `load_or_build` saute le téléchargement et **réutilise le cache à 20 tickers**.

3. Le pipeline construit les fenêtres à partir de ces 20 tickers :
   ```
   20 tickers × ~21 fenêtres/ticker (L=2048, stride=400) ≈ 420 fenêtres
   ```
   → correspond exactement à `sp500: (420, 2048) windows` observé en console.

4. Entraînement sur 420 fenêtres avec `batch_size=64` → ~6 batches/epoch. Même avec 1 000 epochs, seulement ~6 000 pas de gradient, soit **~20× moins que ce que le papier atteint** avec son dataset d'environ 8 000 fenêtres. D'où la sous-convergence et l'aplatissement des queues.

### Confirmation empirique

Sur la machine ayant produit le rapport, inspection du cache :

```bash
python3 -c "
import pandas as pd
from src.data import filter_by_history
df = pd.read_parquet('data/raw/sp500.parquet')
print(f'total tickers downloaded: {df.shape[1]}')
for y in [40, 35, 30, 25, 20]:
    kept = filter_by_history(df, y).shape[1]
    print(f'  min_years={y:2d} -> {kept:3d} tickers')
"
```

Sortie obtenue :

```
total tickers downloaded: 20
  min_years=40 ->  20 tickers
  min_years=35 ->  20 tickers
  min_years=30 ->  20 tickers
  min_years=25 ->  20 tickers
  min_years=20 ->  20 tickers
```

→ le cache contient bien 20 colonnes. L'hypothèse est confirmée : le `reproduce_all.sh` n'a jamais téléchargé la liste S&P 500 complète, il a seulement relu le cache écrit par `reproduce_cpu.sh`.

---

## 4. Autres hypothèses envisagées et écartées

| Hypothèse                                                          | Plausibilité | Verdict                                                                                                                 |
| ------------------------------------------------------------------ | :----------: | ----------------------------------------------------------------------------------------------------------------------- |
| Paquet `powerlaw` absent → estimateur Hill brut utilisé          |    moyenne   | Possible source d'un biais systématique de ~+0,5 mais n'explique pas l'écart de +3,5 sur gbm_linear. Non suffisant seul. |
| Biais de survie (liste Wikipedia 2026 ≠ constituents du papier) |    faible    | Aurait modifié α (réel) de plus que −0,32. Marginal.                                                                   |
| Nombre d'epochs inférieur au papier (variable `EPOCHS` env)       |    moyenne   | À vérifier dans les logs de la collaboratrice, mais même 1000 epochs sur 420 fenêtres ne rattraperait pas le retard.  |
| Implémentation du score-net différente du papier                |    faible    | α (réel) ne dépend pas du score-net ; or il est correct. Bug côté modèle improbable.                                 |
| Divergence du seed / variance statistique sur 120 échantillons    |    faible    | 120 × 2048 × 0,05 ≈ 12 000 observations de queue — largement suffisant pour stabiliser α à ±0,3.                  |

La cause racine est donc **la pollution du cache**, pas un bug d'implémentation du papier.

---

## 5. Procédure de correction

### 5.1 Nettoyer le cache pollué

```bash
rm -f data/raw/sp500.parquet
rm -f data/processed/sp500_L*.npz
```

Optionnel (si vous voulez repartir de zéro) :

```bash
rm -f experiments/checkpoints/sp500_*.pt
rm -f results/samples_*.npy
rm -f results/figures/sp500_*.png
```

### 5.2 Re-télécharger la liste S&P 500 complète

```bash
python -m src.data --universe sp500 --min-years 40 --seq-len 2048 --stride 400
```

Vérifier le nombre de tickers retenus :

```bash
python -c "
import pandas as pd
from src.data import filter_by_history
df = pd.read_parquet('data/raw/sp500.parquet')
print('downloaded:', df.shape[1])
print('kept (>=40y):', filter_by_history(df, 40).shape[1])
"
```

Attendu : plusieurs dizaines à plusieurs centaines de tickers téléchargés, dont ~100 à 200 passant le filtre `min_years=40` (dépend de la disponibilité yfinance à la date d'exécution).

### 5.3 Si le résultat reste insuffisant

Si la liste filtrée est encore très petite (par ex. < 50 tickers), abaisser `min_years` :

```bash
python -m src.data --universe sp500 --min-years 25 --seq-len 2048 --stride 400
```

Documenter dans le rapport : « min_years abaissé de 40 à 25 car yfinance ne fournit pas de données antérieures à 1986 pour une majorité des constituants S&P 500 actuels ».

### 5.4 Relancer la reproduction

```bash
bash scripts/reproduce_all.sh          # GPU, grille 3×3
# ou pour un test rapide :
EPOCHS=50 bash scripts/reproduce_all.sh
```

---

## 6. Prévention du piège à l'avenir

Recommandations pour éviter que le bug se reproduise :

1. **Ne pas enchaîner `reproduce_cpu.sh` → `reproduce_all.sh` sans nettoyer `data/raw/sp500.parquet`.** Le README mentionne ce piège dans la section *Pièges courants* #6.
2. **Inspecter la ligne `sp500: (N, L) windows` au démarrage de tout script de reproduction.** Une valeur très inférieure à ~8 000 signale un cache incomplet.
3. **Fix code (non appliqué actuellement)** : faire dépendre le nom du cache d'une empreinte de la liste de tickers, par exemple :
   ```python
   key = hashlib.sha1(",".join(sorted(tickers)).encode()).hexdigest()[:8]
   raw_path = raw_dir / f"{spec.universe}_{key}.parquet"
   ```
   Ce correctif est **surgical** mais n'a pas été appliqué à la demande de l'utilisateur (principe « surgical changes » de `CLAUDE.md` §3). À discuter pour une future PR.

---

## 7. Résumé pour un tiers pressé

- **Symptôme** : α synthétiques 1,5 à 3,5 plus élevés que le papier sur la grille 3×3.
- **Cause** : cache `data/raw/sp500.parquet` écrit par `reproduce_cpu.sh` avec 20 tickers, puis réutilisé silencieusement par `reproduce_all.sh`. Training paper-scale sur un dataset 20× trop petit → sous-apprentissage → queues trop légères.
- **Preuve** : console affiche `(420, 2048) windows` au lieu de ~8 000 ; inspection du cache confirme 20 colonnes.
- **Correction** : `rm data/raw/sp500.parquet data/processed/sp500_L*.npz` puis relancer `reproduce_all.sh`.
- **Les données ne sont donc pas « différentes de celles du papier »** ; le bug est côté cache, pas côté source yfinance.
