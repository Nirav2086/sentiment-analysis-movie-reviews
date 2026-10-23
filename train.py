"""Train the sentiment model on SST-5 and save it to models/sentiment_model.joblib.

Usage:
    python train.py
    python train.py --output models/my_model.joblib
"""
import argparse

from sklearn.metrics import accuracy_score, classification_report, f1_score

from sentiment_tool import LABELS, MODEL_PATH, build_pipeline, load_data, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sentiment analysis model")
    parser.add_argument("--output", default=str(MODEL_PATH), help="where to save the model")
    args = parser.parse_args()

    print("Loading SST-5 (3-class)...")
    data = load_data()
    train, val, test = data["train"], data["validation"], data["test"]
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    model = build_pipeline()
    model.fit(train["clean"], train["sentiment"])

    val_pred = model.predict(val["clean"])
    print(f"\nValidation accuracy: {accuracy_score(val['sentiment'], val_pred):.3f}")
    print(f"Validation macro F1: {f1_score(val['sentiment'], val_pred, average='macro'):.3f}")

    test_pred = model.predict(test["clean"])
    print("\nTest set report:")
    print(classification_report(test["sentiment"], test_pred, target_names=LABELS, digits=3))

    save_model(model, args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
