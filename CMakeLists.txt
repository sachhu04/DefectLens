cmake_minimum_required(VERSION 3.15)
project(DefectLens NONE)

# Run both backend and frontend concurrently
add_custom_target(run
    COMMAND bash -c "trap 'kill 0' SIGINT SIGTERM EXIT; PYTHONPATH=. uvicorn backend.app.main:app --reload & (cd ${CMAKE_CURRENT_SOURCE_DIR}/frontend && npm run dev) & wait"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Starting DefectLens Backend (FastAPI) and Frontend (Vite)..."
    USES_TERMINAL
)

# Run backend only
add_custom_target(run-backend
    COMMAND uvicorn backend.app.main:app --reload
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Starting DefectLens Backend (FastAPI)..."
    USES_TERMINAL
)

# Run frontend only
add_custom_target(run-frontend
    COMMAND npm run dev
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/frontend
    COMMENT "Starting DefectLens Frontend (Vite)..."
    USES_TERMINAL
)

# Install dependencies
add_custom_target(install-deps
    COMMAND bash -c "pip install -r requirements.txt && pip install -e . && cd frontend && npm install"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Installing Python and Node.js dependencies..."
    USES_TERMINAL
)
