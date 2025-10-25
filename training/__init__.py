# training/train.py
from pathlib import Path

def main():
    print("training entrypoint works ✔")
    # put your real training here, e.g. load data and fit pipeline
    Path("artifacts").mkdir(exist_ok=True)
    (Path("artifacts") / "dummy.txt").write_text("hello")

if __name__ == "__main__":
    main()