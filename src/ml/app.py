from fastapi import FastAPI

app = FastAPI(title="EV ChargeOps ML")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
