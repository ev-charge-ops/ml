from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .serving.registry import load_registry
from .serving.routes import router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_registry()
    yield


app = FastAPI(title="EV ChargeOps ML", version="1.0.0", lifespan=lifespan)
app.include_router(router)
