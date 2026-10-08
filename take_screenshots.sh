#!/bin/bash
cd ~/Desktop/DefectLens

# Command 1: download_dataset
osascript -e 'tell application "Terminal" to do script "cd ~/Desktop/DefectLens && clear && echo \"--- dataset script test ---\" && python scripts/download_dataset.py --help && echo \"\""'
sleep 2
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID screenshot_dataset.png
osascript -e 'tell application "Terminal" to close front window'

# Command 2: tree
osascript -e 'tell application "Terminal" to do script "cd ~/Desktop/DefectLens && clear && echo \"--- directory structure ---\" && find src scripts -type f | sort && echo \"\""'
sleep 2
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID screenshot_tree.png
osascript -e 'tell application "Terminal" to close front window'

# Command 3: mctf.py
osascript -e 'tell application "Terminal" to do script "cd ~/Desktop/DefectLens && clear && echo \"--- MCTF core snippet ---\" && cat src/models/mctf.py | head -n 35 && echo \"\""'
sleep 2
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID screenshot_mctf.png
osascript -e 'tell application "Terminal" to close front window'
