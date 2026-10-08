#!/bin/bash
cd ~/Desktop/DefectLens

osascript -e 'tell application "Terminal"
    set backendTab to do script "cd ~/Desktop/DefectLens && clear && echo \"--- starting FastAPI backend ---\" && /opt/miniconda3/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
    delay 1
    set currentWindow to window 1
    set number of rows of currentWindow to 20
    set number of columns of currentWindow to 100
end tell'
sleep 3
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID -o real_screenshot_backend.png
osascript -e 'tell application "Terminal" to close front window'

osascript -e 'tell application "Terminal"
    set frontendTab to do script "cd ~/Desktop/DefectLens/frontend && clear && echo \"--- starting React Frontend ---\" && npm run dev -- --host 127.0.0.1 --port 5173"
    delay 1
    set currentWindow to window 1
    set number of rows of currentWindow to 20
    set number of columns of currentWindow to 100
end tell'
sleep 4
WINDOW_ID=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID -o real_screenshot_frontend.png
osascript -e 'tell application "Terminal" to close front window'

