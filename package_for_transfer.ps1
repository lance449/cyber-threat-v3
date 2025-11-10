# PowerShell Script to Package Files for Transfer
# This script creates ZIP files of the folders you need to copy

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Packaging Files for Transfer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$basePath = $PSScriptRoot
$outputPath = Join-Path $basePath "transfer_package"

# Create output directory
if (Test-Path $outputPath) {
    Write-Host "Cleaning existing output directory..." -ForegroundColor Yellow
    Remove-Item $outputPath -Recurse -Force
}
New-Item -ItemType Directory -Path $outputPath | Out-Null

Write-Host "📦 Step 1: Packaging Models Folder..." -ForegroundColor Green
$modelsPath = Join-Path $basePath "backend\models"
$modelsZip = Join-Path $outputPath "models.zip"

if (Test-Path $modelsPath) {
    Compress-Archive -Path $modelsPath -DestinationPath $modelsZip -Force
    $modelsSize = (Get-Item $modelsZip).Length / 1MB
    Write-Host "   ✅ Created: models.zip ($([math]::Round($modelsSize, 2)) MB)" -ForegroundColor Green
} else {
    Write-Host "   ❌ ERROR: backend\models folder not found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "📦 Step 2: Packaging Processed Data Folder..." -ForegroundColor Green
$processedPath = Join-Path $basePath "backend\data\processed"
$processedZip = Join-Path $outputPath "processed.zip"

if (Test-Path $processedPath) {
    Compress-Archive -Path $processedPath -DestinationPath $processedZip -Force
    $processedSize = (Get-Item $processedZip).Length / 1MB
    Write-Host "   ✅ Created: processed.zip ($([math]::Round($processedSize, 2)) MB)" -ForegroundColor Green
} else {
    Write-Host "   ❌ ERROR: backend\data\processed folder not found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "📦 Step 3: Packaging CSV Files (Optional)..." -ForegroundColor Green
$consolidatedPath = Join-Path $basePath "backend\data\consolidated"
$csvZip = Join-Path $outputPath "csv_data.zip"

if (Test-Path $consolidatedPath) {
    $csvFiles = Get-ChildItem -Path $consolidatedPath -Filter "*.csv" -File
    if ($csvFiles.Count -gt 0) {
        Compress-Archive -Path "$consolidatedPath\*.csv" -DestinationPath $csvZip -Force
        $csvSize = (Get-Item $csvZip).Length / 1MB
        Write-Host "   ✅ Created: csv_data.zip ($([math]::Round($csvSize, 2)) MB)" -ForegroundColor Green
        Write-Host "   ℹ️  Upload this to online drive for downloading on other device" -ForegroundColor Yellow
    } else {
        Write-Host "   ⚠️  No CSV files found (this is optional)" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ⚠️  backend\data\consolidated folder not found (this is optional)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Packaging Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📁 Output location: $outputPath" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Check the ZIP files in: $outputPath" -ForegroundColor White
Write-Host "   2. Transfer models.zip and processed.zip to new device" -ForegroundColor White
Write-Host "   3. (Optional) Upload csv_data.zip to online drive" -ForegroundColor White
Write-Host "   4. Extract ZIP files to correct locations on new device" -ForegroundColor White
Write-Host ""
Write-Host "   models.zip → Extract to: backend\models\" -ForegroundColor Yellow
Write-Host "   processed.zip → Extract to: backend\data\processed\" -ForegroundColor Yellow
Write-Host "   csv_data.zip → Extract to: backend\data\consolidated\" -ForegroundColor Yellow
Write-Host ""

