$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$q = "neo4j_query_table_data_2026-4-16.json"
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
Write-Host "=== vanilla_qe (126q) ===" -ForegroundColor Cyan
python evaluation.py --questions $q --mode vanilla_qe --llm ollama --write-summary outputs/eval_vanilla_qe.json 2>&1 | Tee-Object -FilePath outputs/log_eval_vanilla_qe_full.txt
if ($LASTEXITCODE -ne 0) { throw "vanilla_qe failed" }
Write-Host "=== kg_rag (126q) ===" -ForegroundColor Cyan
python evaluation.py --questions $q --mode kg_rag --llm ollama --write-summary outputs/eval_kg_rag.json 2>&1 | Tee-Object -FilePath outputs/log_eval_kg_rag_full.txt
if ($LASTEXITCODE -ne 0) { throw "kg_rag failed" }
python aggregate_method_comparison.py --meta-note "neo4j_query_table_data_2026-4-16.json · ollama · full 126 · $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Host "Done." -ForegroundColor Green
