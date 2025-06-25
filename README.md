# Avatar: The Last Airbender - Dialogue Clustering Project

This project applies various **unsupervised machine learning techniques** to analyze and cluster character dialogues from the animated series *Avatar: The Last Airbender*. By transforming text into embeddings and applying dimensionality reduction and clustering algorithms, we aim to reveal semantic patterns in characters' dialogues and episodes.

## 📁 Dataset

- Source: [Kaggle - Avatar: The Last Airbender Dataset](https://www.kaggle.com/datasets/ekrembayar/avatar-the-last-air-bender/data)
- Size: 13,736 rows, 11 columns
- Key fields:
  - `character words`: spoken dialogue
  - `book`, `chapter`: season and episode info
  - `imdb rating`: episode rating

## 🔄 Preprocessing

- **Stopword removal** and **lowercasing** using `nltk`
- Only the `character words` column was used
- No preprocessing applied to numerical fields (`imdb rating`)

## 🔧 Feature Extraction

Two vectorization methods were used:
- **Universal Sentence Encoder (USE)** for semantically rich representations
- **TF-IDF Vectorizer** for capturing keyword importance

After vectorization, **Principal Component Analysis (PCA)** was applied to reduce feature dimensions to 2D for visualization and improved clustering.

## 🔍 Clustering Algorithms

The following unsupervised algorithms were tested:
- **K-Means**
- **Agglomerative Clustering**
- **Spectral Clustering**

The number of clusters (k = 3) was determined using:
- **Elbow Method**
- **Silhouette Score**

## 📊 Evaluation Metrics

- **Intrinsic (unsupervised):**
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
- **Extrinsic (supervised):** (for character clustering)
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)

K-Means performed best overall in terms of interpretability and metric scores.

## 📈 Visualization

- Episode-based and character-based clustering results were visualized using `plotly`
- Word clouds and sentiment analyses for key characters (Aang, Katara, Zuko, etc.)
- IMDB rating distributions per season, episode, and director
- Character co-occurrence graph

## 🧠 Key Learnings

- K-Means provided the most stable clustering results overall
- USE worked better for episode-level clustering, while TF-IDF worked well for character-level
- Spectral clustering was less effective due to smaller sample sizes

## ⚠️ Challenges

- Choosing the best embedding method and clustering algorithm
- Clustering performance varies significantly based on text preprocessing

## 🔮 Future Work

- Test deep learning-based vectorization methods (e.g., BERT, LLM embeddings)
- Apply clustering to scene-level or emotion-based segmentation
- Integrate dialogue context and sequence information

## 📎 Author

- Özge Işıklar  
- TOBB University of Economics and Technology  
- Department of Artificial Intelligence Engineering  
- Email: ozgeisklar@gmail.com

