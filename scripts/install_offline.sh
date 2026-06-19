#!/usr/bin/env bash
#
# install_offline.sh — Installe TabICL GUI sur une machine HORS LIGNE
# à partir des wheels embarquées dans le bundle.
#
# À exécuter DANS le dossier décompressé du bundle (tabicl-gui-offline/).
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

WHEELS_DIR="$ROOT_DIR/wheels"
VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -d "$WHEELS_DIR" ]]; then
  echo "ERREUR : dossier 'wheels/' introuvable. Êtes-vous dans le dossier du bundle ?"
  exit 1
fi

echo "==> Création d'un environnement virtuel : $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Mise à niveau de pip depuis les wheels locales (sans réseau)"
python -m pip install --no-index --find-links "$WHEELS_DIR" --upgrade pip wheel setuptools || true

echo "==> Installation des dépendances (100% hors ligne)"
python -m pip install \
  --no-index \
  --find-links "$WHEELS_DIR" \
  -r requirements.txt

# Placer le checkpoint pré-téléchargé dans le cache attendu, si présent
if compgen -G "$ROOT_DIR/checkpoints/*.ckpt" >/dev/null; then
  CACHE_DIR="$HOME/.cache/tabicl"
  mkdir -p "$CACHE_DIR"
  cp "$ROOT_DIR"/checkpoints/*.ckpt "$CACHE_DIR"/ 2>/dev/null || true
  echo "==> Checkpoint(s) copié(s) dans $CACHE_DIR"
fi

echo ""
echo "============================================================"
echo " Installation terminée."
echo "============================================================"
echo ""
echo "Pour lancer l'application :"
echo "   source $VENV_DIR/bin/activate"
echo "   python app.py"
echo ""
echo "NOTE checkpoint : si TabICL réclame un modèle au lancement,"
echo "indiquez le fichier .ckpt embarqué via le paramètre model_path,"
echo "ou copiez checkpoints/*.ckpt dans le cache TabICL."
