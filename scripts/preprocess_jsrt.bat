@echo off
setlocal

rem Configuration
set "SOURCE_DIR=.\All247images"
set "OUTPUT_DIR=.\processed_data"
set "TRAIN_FILES=train_files.txt"
set "VAL_FILES=val_files.txt"
set "TEST_FILES=test_files.txt"
set TARGET_SIZE="1024" "1024"

rem Print header
echo === JSRT Dataset Preprocessing Pipeline ===
echo Starting execution at %date% %time%

rem Execute the preprocessing script
echo [EXECUTING] Starting JSRT preprocessing...
python jsrt_preprocessing.py ^
    --source-dir "%SOURCE_DIR%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --train-files "%TRAIN_FILES%" ^
    --val-files "%VAL_FILES%" ^
    --test-files "%TEST_FILES%" ^
    --target-size %TARGET_SIZE% ^
    --skip-existing

echo === Processing completed at %date% %time% ===

rem Pause to view results if double-clicked
pause
endlocal