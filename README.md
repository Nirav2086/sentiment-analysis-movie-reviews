# 🎬 Sentiment Analysis Tool for Movie Reviews

A machine learning / NLP project in Python that reads text (movie reviews, tweets) and classifies it as **negative**, **neutral** or **positive**.

Built during my Data Science and Machine Learning internship at YBI Foundation, using Google Colab and Jupyter Notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sentiment-analysis-movie-reviews/blob/main/notebooks/sentiment_analysis_movie_reviews.ipynb)

## Features

- 3-class sentiment (negative / neutral / positive) from the SST-5 movie-review dataset
- Text cleaning that keeps negation words ("not good" stays negative)
- TF-IDF (unigrams + bigrams) with Logistic Regression, compared against Linear SVM and Naive Bayes
- Confidence score and per-class probabilities for every prediction
- Interactive widget in the notebook, a command-line tool and a Gradio web demo
- Optional comparison with a pretrained RoBERTa transformer

## How it works

```
text -> clean_text() -> TF-IDF vectorizer -> Logistic Regression -> label + confidence
```

SST-5 has five labels. They are mapped to three classes:

| Original label | Meaning | Class |
|---|---|---|
| 0, 1 | very negative, negative | negative |
| 2 | neutral | neutral |
| 3, 4 | positive, very positive | positive |

## Project structure

```
sentiment-analysis-movie-reviews/
├── notebooks/
│   └── sentiment_analysis_movie_reviews.ipynb   # full walkthrough (run in Colab)
├── sentiment_tool.py    # cleaning, training pipeline, predict_sentiment()
├── train.py             # train + evaluate + save the model
├── predict.py           # command-line predictions
├── app.py               # Gradio web demo
├── models/              # saved model goes here (ignored by git)
├── requirements.txt
└── LICENSE
```

## Quick start

### Option 1: Google Colab (easiest)

Click the **Open in Colab** badge above and run the cells from top to bottom. Switch to a GPU runtime for the transformer section.

### Option 2: Run locally

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-movie-reviews.git
cd sentiment-analysis-movie-reviews

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py                  # trains, prints the test report, saves the model
python predict.py "An absolute masterpiece. The acting was breathtaking!"
python app.py                    # opens the web demo
```

## Example

```python
from sentiment_tool import load_model, predict_sentiment

model = load_model()
predict_sentiment([
    "An absolute masterpiece!",
    "Boring, predictable, and way too long.",
    "Not bad, but not great either.",
], model)
```

Returns a table with `sentiment`, `confidence`, `p_negative`, `p_neutral` and `p_positive` for each text.

## Results

Run `python train.py` (or the notebook) and add your numbers here:

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | _fill in_ | _fill in_ |
| TF-IDF + Linear SVM | _fill in_ | _fill in_ |
| TF-IDF + Naive Bayes | _fill in_ | _fill in_ |
| Pretrained RoBERTa (500-review sample) | _fill in_ | _fill in_ |

## Limitations

- Neutral is the hardest class: it is the smallest and overlaps with mildly positive and negative text.
- TF-IDF models see words, not deep context, so sarcasm and long-range negation can be misclassified.
- The model is trained on movie-review sentences, so accuracy on very different text (for example tweets) may drop.

## Future work

- Fine-tune DistilBERT on SST-5
- Test on a tweet dataset (`tweet_eval`, `sentiment`)
- Deploy the Gradio app on Hugging Face Spaces

## Tech stack

Python, pandas, NumPy, scikit-learn, Hugging Face datasets and transformers, Gradio, Google Colab.

## Dataset and credits

- SST-5: Socher et al., *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*, EMNLP 2013 (loaded from Hugging Face `SetFit/sst5`)
- Pretrained model: `cardiffnlp/twitter-roberta-base-sentiment-latest`

## Author

**Pavan Dharmendra Upadhyay**, Computer Engineering, K. J. Institute of Engineering and Technology, Savli, Vadodara

## License

MIT, see [LICENSE](LICENSE).
