#!/usr/bin/env bash
# Lance l'API FastAPI (uvicorn) sur le port 8000.
#
# Prérequis :
#   - Dépendances Python installées (cf. requirements.txt)
#   - Variable d'environnement OPENAI_API_KEY définie (dans .env à la racine ou
#     exportée dans le shell) ; sans ça, POST /ask renverra une erreur 503.
#
# Ctrl+C pour arrêter.

set -euo pipefail

# Se placer à la racine du repo (là où se trouve ce script), quel que soit
# l'endroit d'où on l'appelle.
cd "$(dirname "$0")"

exec python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
