$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$q = "neo4j_query_table_data_2026-4-16.json"
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
$runs = @(
  @("no_retrieval", "outputs/eval_no_retrieval.json"),
  @("vanilla_rag", "outputs/eval_vanilla_rag.json"),
  @("vanilla_qe", "outputs/eval_vanilla_qe.json"),
  @("kg_rag", "outputs/eval_kg_rag.json")
)
foreach ($r in $runs) {
  $mode = $r[0]
  $out = $r[1]
  Write-Host "=== evaluation.py --mode $mode ===" -ForegroundColor Cyan
  python evaluation.py --questions $q --mode $mode --llm ollama --write-summary $out 2>&1 | Tee-Object -FilePath "outputs/log_eval_$mode.txt"
  if ($LASTEXITCODE -ne 0) { throw "evaluation failed for $mode exit $LASTEXITCODE" }
}
python aggregate_method_comparison.py --meta-note "neo4j_query_table_data_2026-4-16.json · ollama · full run $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
if ($LASTEXITCODE -ne 0) { throw "aggregate_method_comparison failed" }
Write-Host "Done. method_comparison.json updated." -ForegroundColor Green
