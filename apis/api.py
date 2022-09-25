from fastapi import FastAPI

from apis.routes.graph_translate import GraphTranslateRoute
from apis.routes.translation import TranslateRoute


app = FastAPI()
app.include_router(GraphTranslateRoute().router)
app.include_router(TranslateRoute().router)
