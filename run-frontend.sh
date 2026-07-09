#!/usr/bin/env bash
# Lance le serveur de dev Vite (frontend React) sur le port 5173.
#
# Prérequis :
#   - Node.js installé (v18+)
#   - `npm install` déjà effectué dans frontend/
#
# Ctrl+C pour arrêter.

set -euo pipefail

cd "$(dirname "$0")/frontend"

exec npm run dev
