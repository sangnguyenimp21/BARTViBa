from fastapi import FastAPI

from apis.routes.graph_translate import GraphTranslateRoute
from apis.routes.translation import TranslateRoute

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(GraphTranslateRoute().router)
app.include_router(TranslateRoute().router)
