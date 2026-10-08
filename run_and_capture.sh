#!/bin/bash
cd ~/Desktop/DefectLens

# Command 1: download_dataset.py
osascript -e 'tell application "Terminal"
    do script "cd ~/Desktop/DefectLens && clear && echo \"--- dataset script test ---\" && python scripts/download_dataset.py --help && touch /tmp/done_dataset"
    set number of rows of front window to 20
    set number of columns of front window to 100
end tell'
while [ ! -f /tmp/done_dataset ]; do sleep 1; done
sleep 1
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID -o real_screenshot_dataset.png
osascript -e 'tell application "Terminal" to close front window'
rm -f /tmp/done_dataset

# Command 2: tree / find
osascript -e 'tell application "Terminal"
    do script "cd ~/Desktop/DefectLens && clear && echo \"--- directory structure ---\" && find src scripts data/mvtec/bottle/ -type d | sort | head -n 25 && touch /tmp/done_tree"
    set number of rows of front window to 30
    set number of columns of front window to 100
end tell'
while [ ! -f /tmp/done_tree ]; do sleep 1; done
sleep 1
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID -o real_screenshot_tree.png
osascript -e 'tell application "Terminal" to close front window'
rm -f /tmp/done_tree

# Command 3: mctf.py
osascript -e 'tell application "Terminal"
    do script "cd ~/Desktop/DefectLens && clear && echo \"--- MCTF core snippet ---\" && cat src/models/mctf.py | head -n 42 && touch /tmp/done_mctf"
    set number of rows of front window to 45
    set number of columns of front window to 110
end tell'
while [ ! -f /tmp/done_mctf ]; do sleep 1; done
sleep 1
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID -o real_screenshot_mctf.png
osascript -e 'tell application "Terminal" to close front window'
rm -f /tmp/done_mctf

