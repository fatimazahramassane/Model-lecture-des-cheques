# Étude de cas 1 – Jalon 3 : Un nouveau scanner, deux tâches

## Contenu
- `uc1_jalon3_v7.ipynb` : notebook complet (autonome, graine fixée à 42).
- `journal_jalon3.jsonl` / `journal_jalon3.csv` : journal d'expériences.
- `lecteur_A.keras`, `lecteur_A_renforce.keras` : lecteurs.
- `multitache_A.keras`, `multitache_B_final.keras`, `multitache_B_graine1/2.keras` : modèle multitâche A et modèles adaptés à B (ensemble de 3).
- `resultats_multitache.csv`, `tableau_strategies.csv`, `tableau_B_600.csv`, `bilan_test.csv`.
- `requirements.txt` : versions utilisées ({"python": "3.13.16", "numpy": "2.1.3", "pandas": "2.2.3", "scipy": "1.16.3", "scikit-learn": "1.6.1", "matplotlib": "3.10.0", "tensorflow": "2.21.0", "keras": "3.13.2"}).

## Données
- `uc1_domaineA.npz`, `uc1_domaineB.npz` dans `data/`.

## Exécution
1. `pip install -r requirements.txt`.
2. Colab GPU recommandé.
3. Exécuter les cellules dans l'ordre (1 h 15 à 2 h avec GPU).
4. Sorties dans `/content/drive/MyDrive/uc1_jalon3/sorties_jalon3/`.
