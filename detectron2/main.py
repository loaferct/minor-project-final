import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from classify_api import router as classify_router
# from PaddleOCR.api_test import router as table_router
from PaddleOCR.table_ocr_api import router as table_router
from PaddleOCR.piechart_api import router as pie_router
from PaddleOCR.pie_paddle import router as pie_paddle
from PaddleOCR.bar_paddle_api import router as bar_paddle
from bar_boudning_api import router as bar_bounding
# from bar_reconsturct_api import router as bar_reconstruct

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
app.include_router(pie_router, prefix="/pie", tags=["Pie Reconstruct"])
app.include_router(pie_paddle, prefix="/pie", tags=["Pie Reconstruct"])
app.include_router(bar_paddle, prefix="/bar", tags=["Bar Reconstruct"])
app.include_router(bar_bounding, prefix="/bar", tags=["Bar Reconstruct"])
# app.include_router(bar_reconstruct, prefix="/bar", tags=["Bar Reconstruct"])