#!/bin/bash
cd ~/Desktop/DefectLens

# Command 1: predict.py (Explicitly use conda python to avoid ModuleNotFoundError)
osascript -e 'tell application "Terminal"
    set newTab to do script "cd ~/Desktop/DefectLens && clear && echo \"--- anomaly prediction ---\" && /opt/miniconda3/bin/python scripts/predict.py --image data/mvtec/bottle/test/broken_small/000.png --model mctf && touch /tmp/done_predict"
    delay 0.5
    set currentWindow to window 1
    set number of rows of currentWindow to 25
    set number of columns of currentWindow to 100
end tell'
count=0
while [ ! -f /tmp/done_predict ] && [ $count -lt 30 ]; do sleep 1; count=$((count+1)); done
sleep 2 
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID -o real_screenshot_predict.png
osascript -e 'tell application "Terminal" to close front window'
rm -f /tmp/done_predict

