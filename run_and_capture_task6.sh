#!/bin/bash
cd ~/Desktop/DefectLens

# Command 1: backend server
osascript -e 'tell application "Terminal"
    set backendWindow to do script "cd ~/Desktop/DefectLens && clear && echo \"--- starting FastAPI backend ---\" && /opt/miniconda3/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 && touch /tmp/done_backend_server"
    delay 0.5
    set number of rows of window 1 to 20
    set number of columns of window 1 to 100
end tell'
sleep 3
WINDOW_ID_BACKEND=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID_BACKEND -o real_screenshot_backend.png

# Command 2: frontend server
osascript -e 'tell application "Terminal"
    set frontendWindow to do script "cd ~/Desktop/DefectLens/frontend && clear && echo \"--- starting React Frontend ---\" && npm run dev -- --host 127.0.0.1 --port 5173"
    delay 0.5
    set number of rows of window 1 to 20
    set number of columns of window 1 to 100
end tell'
sleep 5 # wait for vite to start
WINDOW_ID_FRONTEND=$(osascript -e 'tell application "Terminal" to get id of front window')
screencapture -l $WINDOW_ID_FRONTEND -o real_screenshot_frontend.png

