from fastapi import FastAPI
from services import router

app = FastAPI()

# Include recommendation routes
app.include_router(router)
