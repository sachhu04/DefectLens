#!/bin/bash
set -e
cd ~/Desktop/DefectLens

echo "=== 1. Capturing Integration Test Suite ==="
rm -f /tmp/done_test
osascript <<EOF
tell application "Terminal"
    activate
    set w to do script "cd ~/Desktop/DefectLens && clear && echo \"--- Task 7: Full System Module Integration Test ---\" && export KMP_DUPLICATE_LIB_OK=TRUE && /opt/miniconda3/bin/python scripts/test_pipeline.py && touch /tmp/done_test"
    set number of rows of window 1 to 24
    set number of columns of window 1 to 95
end tell
EOF

while [ ! -f /tmp/done_test ]; do sleep 1; done
sleep 2
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of window 1')
screencapture -l $WINDOW_ID -o real_screenshot_integration.png
osascript -e 'tell application "Terminal" to close window 1'
rm -f /tmp/done_test
echo "Saved real_screenshot_integration.png"

echo "=== 2. Capturing Benchmark Evaluation ==="
rm -f /tmp/done_eval
osascript <<EOF
tell application "Terminal"
    activate
    set w to do script "cd ~/Desktop/DefectLens && clear && echo \"--- Task 7: Quantitative Benchmark Evaluation (MCTF ViT) ---\" && export KMP_DUPLICATE_LIB_OK=TRUE && /opt/miniconda3/bin/python scripts/evaluate.py --config configs/mctf.yaml --category bottle && touch /tmp/done_eval"
    set number of rows of window 1 to 20
    set number of columns of window 1 to 95
end tell
EOF

while [ ! -f /tmp/done_eval ]; do sleep 1; done
sleep 2
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of window 1')
screencapture -l $WINDOW_ID -o real_screenshot_evaluate.png
osascript -e 'tell application "Terminal" to close window 1'
rm -f /tmp/done_eval
echo "Saved real_screenshot_evaluate.png"

echo "=== 3. Capturing Single Inference ==="
rm -f /tmp/done_pred
osascript <<EOF
tell application "Terminal"
    activate
    set w to do script "cd ~/Desktop/DefectLens && clear && echo \"--- Task 7: Anomaly Inspection Inference Engine ---\" && export KMP_DUPLICATE_LIB_OK=TRUE && /opt/miniconda3/bin/python scripts/predict.py --image data/mvtec/bottle/test/broken_small/000.png --model mctf && touch /tmp/done_pred"
    set number of rows of window 1 to 18
    set number of columns of window 1 to 95
end tell
EOF

while [ ! -f /tmp/done_pred ]; do sleep 1; done
sleep 2
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of window 1')
screencapture -l $WINDOW_ID -o real_screenshot_predict.png
osascript -e 'tell application "Terminal" to close window 1'
rm -f /tmp/done_pred
echo "Saved real_screenshot_predict.png"

echo "=== 4. Capturing Git Delivery & Repository Sync ==="
rm -f /tmp/done_git
osascript <<EOF
tell application "Terminal"
    activate
    set w to do script "cd ~/Desktop/DefectLens && clear && echo \"--- Task 7: Git Repository Delivery & Remote Sync ---\" && git log -n 5 --oneline && echo \"\" && git remote -v && touch /tmp/done_git"
    set number of rows of window 1 to 16
    set number of columns of window 1 to 95
end tell
EOF

while [ ! -f /tmp/done_git ]; do sleep 1; done
sleep 2
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of window 1')
screencapture -l $WINDOW_ID -o real_screenshot_git.png
osascript -e 'tell application "Terminal" to close window 1'
rm -f /tmp/done_git
echo "Saved real_screenshot_git.png"

echo "=== ALL SCREENSHOTS SUCCESSFULLY CAPTURED AND VERIFIED ==="
