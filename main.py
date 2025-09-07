from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import users, diet, chatbot,recipe

app = FastAPI(title="DangDangFit API", version="0.1.0")


# CORS (필요 시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
# app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(diet.router, prefix="/api/model", tags=["Diet"])
app.include_router(recipe.router, prefix="/api/model", tags=["Recipe"])
app.include_router(chatbot.router, prefix="/api/model", tags=["Chatbot"])

@app.get("/")
def health_check():
    return {"status": "ok", "message": "DangDangFit API running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=8000)