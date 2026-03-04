#!/usr/bin/env python3
# starts the fastapi retrieval server on port 8000
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.retrieval_api:app", host="0.0.0.0", port=8000, reload=True)
