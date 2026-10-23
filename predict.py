"""Command-line sentiment prediction.

Usage:
    python predict.py "An absolute masterpiece!"
    python predict.py            # interactive mode (type reviews, Ctrl+C to quit)
"""
import sys

from sentiment_tool import MODEL_PATH, load_model, predict_sentiment


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
    model = load_model()

    if len(sys.argv) > 1:
        print(predict_sentiment(" ".join(sys.argv[1:]), model).to_string(index=False))
        return

    print("Type a review and press Enter (Ctrl+C to quit).")
    try:
        while True:
            text = input("> ").strip()
            if text:
                print(predict_sentiment(text, model).iloc[0][["sentiment", "confidence"]].to_string(), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    main()
