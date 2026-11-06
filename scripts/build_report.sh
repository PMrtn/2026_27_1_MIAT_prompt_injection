#!/usr/bin/env sh
# Build docs/report/report.pdf from docs/report/report.md with pandoc + tectonic.
# pandoc comes from PATH or, failing that, from the pypandoc_binary package of the project venv.
set -eu
cd "$(dirname "$0")/../docs/report"
PANDOC="${PANDOC:-$(command -v pandoc || python -c 'import pypandoc; print(pypandoc.get_pandoc_path())')}"
"$PANDOC" report.md -o report.pdf --pdf-engine=tectonic
echo "Wrote docs/report/report.pdf"
