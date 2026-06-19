#!/usr/bin/env bash
#
# bundle_offline.sh — Fabrique une archive autonome de TabICL GUI
# pour installation sur une machine HORS LIGNE.
#
# À exécuter sur une machine CONNECTÉE à Internet.
#
# Le bundle contient :
#   - le code source de l'application
#   - toutes les dépendances Python sous forme de wheels (.whl / .tar.gz)
#   - les scripts d'installation hors ligne
#
# Par défaut, les wheels sont téléchargées pour la plateforme COURANTE.
# Si la machine cible est différente, surchargez via les variables :
#   TARGET_PLATFORM   (ex: manylinux2014_x86_64, win_amd64, macosx_11_0_arm64)
#   TARGET_PYTHON     (ex: 311 pour Python 3.11)
#
# Exemples :
#   ./scripts/bundle_offline.sh
#   TARGET_PLATFORM=win_amd64 TARGET_PYTHON=311 ./scripts/bundle_offline.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BUNDLE_NAME="tabicl-gui-offline"
BUNDLE_DIR="$ROOT_DIR/$BUNDLE_NAME"
WHEELS_DIR="$BUNDLE_DIR/wheels"

echo "==> Nettoyage de l'ancien bundle"
rm -rf "$BUNDLE_DIR"
mkdir -p "$WHEELS_DIR"

echo "==> Copie du code source de l'application"
cp -r app.py requirements.txt README.md tabicl_gui "$BUNDLE_DIR/"
cp scripts/install_offline.sh "$BUNDLE_DIR/"
chmod +x "$BUNDLE_DIR/install_offline.sh"

echo "==> Mise à jour de pip/wheel (utiles pour le téléchargement)"
python -m pip install --upgrade pip wheel >/dev/null

# Construire les arguments de plateforme si une cible spécifique est demandée
PLATFORM_ARGS=()
if [[ -n "${TARGET_PLATFORM:-}" ]]; then
  PLATFORM_ARGS+=(--platform "$TARGET_PLATFORM" --only-binary=:all:)
  echo "==> Cible plateforme : $TARGET_PLATFORM"
fi
if [[ -n "${TARGET_PYTHON:-}" ]]; then
  PLATFORM_ARGS+=(--python-version "$TARGET_PYTHON")
  echo "==> Cible Python : $TARGET_PYTHON"
fi

echo "==> Téléchargement de toutes les dépendances (wheels) dans $WHEELS_DIR"
echo "    (cela peut prendre plusieurs minutes — torch est volumineux)"
python -m pip download \
  -r requirements.txt \
  -d "$WHEELS_DIR" \
  "${PLATFORM_ARGS[@]}"

# On embarque aussi pip lui-même pour pouvoir l'installer hors ligne si besoin
python -m pip download pip wheel setuptools -d "$WHEELS_DIR" "${PLATFORM_ARGS[@]}" || true

echo "==> Tentative de pré-téléchargement du checkpoint TabICL"
# Le modèle est sinon téléchargé au 1er usage, ce qui échoue hors ligne.
mkdir -p "$BUNDLE_DIR/checkpoints"
if python -c "import tabicl" 2>/dev/null; then
  python - "$BUNDLE_DIR/checkpoints" <<'PY' || echo "    (échec — voir la note dans README, à faire manuellement)"
import shutil, sys, glob, os
dest = sys.argv[1]
try:
    from tabicl import TabICLClassifier
    clf = TabICLClassifier()  # déclenche la résolution/téléchargement du checkpoint
    # Localiser le cache des checkpoints (~/.cache ou dossier du paquet)
    candidates = []
    home = os.path.expanduser("~")
    for pat in ("**/tabicl*classifier*.ckpt", "**/tabicl*regressor*.ckpt"):
        candidates += glob.glob(os.path.join(home, ".cache", "**", pat), recursive=True)
    import tabicl as _t
    pkg = os.path.dirname(_t.__file__)
    for pat in ("**/*.ckpt",):
        candidates += glob.glob(os.path.join(pkg, pat), recursive=True)
    seen = set()
    for c in candidates:
        if c not in seen and os.path.isfile(c):
            shutil.copy(c, dest)
            seen.add(c)
            print(f"    checkpoint copié : {os.path.basename(c)}")
    if not seen:
        print("    aucun checkpoint trouvé automatiquement")
except Exception as e:
    print(f"    impossible de pré-télécharger : {e}")
PY
else
  echo "    tabicl n'est pas installé sur cette machine — checkpoint non embarqué."
  echo "    (Le bundle reste valide ; voir la note checkpoint dans le README.)"
fi

echo "==> Création de l'archive"
ARCHIVE="$ROOT_DIR/${BUNDLE_NAME}.tar.gz"
rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" -C "$ROOT_DIR" "$BUNDLE_NAME"

SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo ""
echo "============================================================"
echo " Bundle créé : $ARCHIVE  ($SIZE)"
echo "============================================================"
echo ""
echo "Transférez ce fichier sur la machine hors ligne, puis :"
echo "   tar -xzf ${BUNDLE_NAME}.tar.gz"
echo "   cd ${BUNDLE_NAME}"
echo "   ./install_offline.sh"
