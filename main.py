from pathlib import Path
import sys
import uvicorn

def main():
    # Para adicionar os diretórios de src/ ao path
    ROOT = Path(__file__).resolve().parent
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8181,
        reload=True
    )

if __name__ == "__main__":
    main()