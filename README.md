# GPS Dashboard (Locus Map GPX)

Application Streamlit locale pour analyser des trajets GPS (voiture / marche) à partir d'un fichier GPX.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

## Fonctionnalités

- Upload GPX
- Calculs : vitesse, distance cumulée, durée
- Carte interactive avec coloration par vitesse
- Graphiques vitesse/temps, distance/temps, histogramme de vitesses
- Détection des pauses / arrêts
- Identification des segments rapides/lents et zones lentes
- Heatmap du trajet
