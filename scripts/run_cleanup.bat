@echo off
REM ============================================================
REM  FishVision-AI  --  Repository Cleanup Script
REM  Run from project root: c:\fish_detection
REM  Usage: scripts\run_cleanup.bat
REM ============================================================

cd /d c:\fish_detection

echo.
echo ============================================================
echo  FishVision-AI Repository Cleanup
echo ============================================================
echo.

REM ── 1. Create archive directories ────────────────────────────
echo [1/9] Creating archive directories...
mkdir archive\legacy\classification\trainer   2>nul
mkdir archive\legacy\classification\dataloader 2>nul
mkdir archive\legacy\classification\metrics   2>nul
mkdir archive\legacy\classification\augmentation 2>nul
mkdir archive\legacy\classification\config    2>nul
mkdir archive\experiments                     2>nul
mkdir scripts\debug                           2>nul
echo     Done.

REM ── 2. Remove __pycache__ (auto-regenerated) ─────────────────
echo [2/9] Removing __pycache__ folders...
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" (
        if not "%%d"=="%cd%\venv\__pycache__" (
            rd /s /q "%%d"
            echo     Removed: %%d
        )
    )
)
del /s /q *.pyc 2>nul
echo     Done.

REM ── 3. Remove swap and temp files ────────────────────────────
echo [3/9] Removing swap files...
del /s /q "OzFish\frame_metadata.csv.crswap" 2>nul
del /s /q *.crswap 2>nul
del /s /q *.swp   2>nul
echo     Done.

REM ── 4. Remove personal file from assets ──────────────────────
echo [4/9] Removing personal files...
del "assets\Student Profile @ GTU.pdf" 2>nul
echo     Done.

REM ── 5. Remove generated cache ────────────────────────────────
echo [5/9] Removing generated dataset cache...
del "datasets\cache\parsed_metadata.parquet" 2>nul
echo     Done.

REM ── 6. Remove duplicate root yolov8n.pt ──────────────────────
echo [6/9] Removing duplicate root yolov8n.pt (kept in models\)...
del "yolov8n.pt" 2>nul
echo     Done.

REM ── 7. Remove local venv (project uses Conda) ────────────────
echo [7/9] Removing venv\ (Conda ai env is used instead)...
rd /s /q venv 2>nul
echo     Done.

REM ── 8. Archive legacy code ───────────────────────────────────
echo [8/9] Archiving proven-dead legacy files...

REM bioclip_trainer.py — uses old transformers API, zero imports
move "classification\trainer\bioclip_trainer.py" "archive\legacy\classification\trainer\" 2>nul
echo     Archived: classification\trainer\bioclip_trainer.py

REM bioclip_dataset.py — superseded by fish_dataset.py, zero imports
move "classification\dataloader\bioclip_dataset.py" "archive\legacy\classification\dataloader\" 2>nul
echo     Archived: classification\dataloader\bioclip_dataset.py

REM classification_metrics.py — only used by bioclip_trainer.py
move "classification\metrics\classification_metrics.py" "archive\legacy\classification\metrics\" 2>nul
echo     Archived: classification\metrics\classification_metrics.py

REM fish_augmentation.py — zero imports anywhere
move "classification\augmentation\fish_augmentation.py" "archive\legacy\classification\augmentation\" 2>nul
echo     Archived: classification\augmentation\fish_augmentation.py

REM default.yaml — zero Python imports, superseded by dataset.yaml
move "classification\config\default.yaml" "archive\legacy\classification\config\" 2>nul
echo     Archived: classification\config\default.yaml

echo     Done.

REM ── 9. Move debug scripts and batch file ─────────────────────
echo [9/9] Moving debug scripts to scripts\debug\...
move "scripts\debug_filenames.py" "scripts\debug\" 2>nul
move "scripts\test_dataset.py"    "scripts\debug\" 2>nul
move "run_phase2.bat"             "scripts\"        2>nul
echo     Done.

REM ── 10. Archive old experiments (keep 2 latest) ──────────────
echo [10/10] Archiving old experiments...

set KEEP1=bioclip2_full_20260716_160143
set KEEP2=bioclip2_full_20260715_235014

for /d %%d in (experiments\*) do (
    set "dname=%%~nd"
    if not "%%~nd"=="%KEEP1%" if not "%%~nd"=="%KEEP2%" (
        move "%%d" "archive\experiments\" >nul 2>&1
        echo     Archived: %%~nd
    )
)
echo     Done.

echo.
echo ============================================================
echo  CLEANUP COMPLETE
echo ============================================================
echo.
echo  Kept experiments:
echo    - experiments\%KEEP1%
echo    - experiments\%KEEP2%
echo.
echo  Next: Run a quick import check:
echo    C:\ProgramData\miniconda3\envs\ai\python.exe scripts\verify_backbone.py
echo.
pause
