# Genre-Aware BERT4Rec for MovieLens 100k

This project extends the BERT4Rec-VAE-PyTorch model to provide **genre-specific recommendations** using the **MovieLens 100k dataset**.

## 📌 Features
- Incorporates genre data into input processing.
- Generates **top-5 recommendations per genre** for each user.
- Computes and reports **Recall@5** per genre and overall.
- Includes final results and saved outputs in JSON format.

## 📁 Included Files
- Modified code in:
  - `models/bert_genre.py`
  - `trainers/bert_genre.py`
  - `dataloaders/bert_genre.py`
  - Other updated `__init__.py`, `templates.py`, `options.py`, and `main.py`.
- Jupyter Notebook:
  - `GenreBERT4Rec_Evaluation.ipynb` — to demonstrate training, evaluation, and recommendations.
- Results:
  - `genre_recommendations.json`
  - `genre_recall_results.json`
  - `final_recall_metrics.json`

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Set up the environment and run the notebook
