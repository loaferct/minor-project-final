import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from classify_api import router as classify_router
# from PaddleOCR.api_test import router as table_router
from PaddleOCR.table_ocr_api import router as table_router

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Group routers
app.include_router(classify_router, prefix="/classification", tags=["Classification"])
app.include_router(table_router, prefix="/table", tags=["Table Recognition"])

