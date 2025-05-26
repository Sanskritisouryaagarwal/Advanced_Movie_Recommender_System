# 🎬 Advanced_Movie_Recommender_System

A hybrid machine learning-based movie recommender system that combines **Content-Based Filtering**, **Collaborative Filtering**, and **Matrix Factorization** using SVD. It delivers personalized recommendations based on user preferences and metadata, and is deployed with a responsive web interface using Flask.

---

## 📖 Introduction

With the vast availability of digital content, users often face the problem of information overload. Recommender systems help mitigate this issue by **filtering and ranking items** based on user preferences and item characteristics. The objective of this project is to **simulate a real-world recommendation engine**, optimized for both performance and scalability.

This system blends the best of:
- 📘 Content-Based Filtering: Focused on **item similarity** using metadata
- 👥 Collaborative Filtering: Focused on **user behavior** and **community preferences**
- 🧠 Model-Based Learning: Extracting **latent features** using **Singular Value Decomposition (SVD)**

---

## 📌 Key Features

- 🔁 **Hybrid Recommendation Engine** combining CF and CBF
- 🧮 **Matrix Factorization** for dimensionality reduction and sparsity handling
- 🧾 **Enriched metadata** from APIs like TMDB and OMDB
- ⚡ **Real-time prediction** via a Flask-based backend
- 🌐 **Fully responsive web interface** for movie input and recommendations
- ☁️ **Deployed using Heroku**, ready for real-world use

---

## 🎯 Objective

To design and develop an **intelligent, scalable, and explainable recommendation system** using hybrid machine learning techniques that can:

- Suggest unseen movies to users
- Adapt to new users/items (cold-start problem)
- Handle data sparsity and real-world scalability issues
- Provide reliable and personalized results with minimal latency

---

## 🧠 Theoretical Concepts

### 1. Recommender Systems

A **Recommender System** is an algorithmic system that filters and suggests items to users based on various data inputs. These systems are critical to platforms like Netflix, Amazon, and Spotify.

**Categories:**

| Type                    | Basis                                    | Pros                            | Cons                          |
|-------------------------|------------------------------------------|----------------------------------|-------------------------------|
| Content-Based Filtering | Similarity between items                 | Works for new users             | Cannot discover new interests |
| Collaborative Filtering | Similarity between users/items based on interactions | Learns user patterns          | Cold-start & sparsity issues  |
| Hybrid                  | Combines both above                      | Accurate, cold-start tolerant   | More complex to implement     |

---

## 📊 Dataset

- **MovieLens 100K Dataset**
- 📈 100,000 ratings from 943 users across 1,682 movies
- Fields: `user_id`, `item_id`, `rating`, `timestamp`
- Extended with: `title`, `genre`, `director`, `actors` (via TMDB/OMDB API)

---

## 🧹 Data Preprocessing

1. **Merge Rating and Metadata** using `item_id`
2. Handle **missing values** and format inconsistencies
3. Create **User-Item interaction matrix** (`943 x 1682`)
4. **Vectorize** textual metadata using:
   - `TfidfVectorizer` for plot and genres
   - `CountVectorizer` for categorical data

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
````

---

## 🔍 Recommendation Techniques

### 📘 Content-Based Filtering (CBF)

* Each item is represented as a **feature vector** of content attributes.
* **Cosine Similarity** is used to find closeness between items.
* For a given movie, find others with **similar metadata**.

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(tfidf_matrix)
```

> Suitable for new users with little to no interaction data.

---

### 🤝 Collaborative Filtering (CF)

#### Memory-Based CF

* Uses **raw rating data** to calculate user/item similarities.
* **User-User Filtering**: "Users similar to you liked..."
* **Item-Item Filtering**: "Users who liked this also liked..."

#### Model-Based CF (SVD)

* Performs **matrix factorization** using SVD.
* Projects data into **latent feature space**, learns hidden patterns.
* Reduces overfitting and improves generalization.

```python
from scipy.sparse.linalg import svds
U, S, Vt = svds(train_data_matrix, k=20)
S_diag = np.diag(S)
pred_ratings = np.dot(np.dot(U, S_diag), Vt)
```

> Works well with sparse matrices and captures complex relationships.

---

## 🧬 Hybrid Recommendation System

* **Combines** scores from content and collaborative models
* Use **weighted average** or **ranking fusion**
* Allows fallback to metadata when interaction data is missing
* Helps reduce **cold-start and sparsity issues**

---

## ⚙️ Model Evaluation

### 📐 Metric: RMSE

RMSE = √(∑(y\_true - y\_pred)² / N)

```python
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(y_true, y_pred))
```

| Model                | RMSE |
| -------------------- | ---- |
| User-User CF         | 3.13 |
| Item-Item CF         | 3.46 |
| Model-Based CF (SVD) | 2.72 |

> Lower RMSE = Better prediction accuracy

---

## 📊 Sparsity Analysis

Most real-world datasets are **sparse**. This means only a few ratings are known:

```python
sparsity = 1.0 - len(df) / float(n_users * n_items)
print(f"Sparsity: {sparsity*100:.2f}%")
```

**Sparsity: 93.7%**

> High sparsity motivates the need for model-based approaches like SVD.

---

## 🌐 Web Interface

### Built With:

* Flask (Python)
* HTML, CSS, JavaScript
* Bootstrap for styling

### Features:

* 🔍 Movie search bar
* 🎥 Top-N recommendation list
* 🖼️ Poster preview using TMDB API
* 🔄 Refreshable output area

---

## ☁️ Deployment

| Method      | Platform   | Details                                   |
| ----------- | ---------- | ----------------------------------------- |
| Cloud       | Heroku     | Hosted Flask app with GitHub integration  |
| Container   | Docker     | Containerized for portability and scaling |
| Cloud Infra | AWS EC2/S3 | Deployed via custom infrastructure setup  |

---

## 💻 Tech Stack Overview

| Category           | Tools/Technologies                    |
| ------------------ | ------------------------------------- |
| Language           | Python                                |
| Libraries          | NumPy, Pandas, Scikit-learn, Flask    |
| Text Vectorization | TfidfVectorizer, CountVectorizer      |
| Similarity Metrics | Cosine Similarity, Euclidean Distance |
| Frontend           | HTML5, CSS3, Bootstrap, JS            |
| APIs               | TMDB, OMDB                            |
| Deployment         | Docker, Heroku, AWS                   |
| Version Control    | Git, GitHub                           |

---

## 📈 Learning Outcomes

* ✅ Designed and implemented multiple types of recommender systems
* ✅ Understood limitations of CF and how hybrid systems overcome them
* ✅ Handled real-world data issues like sparsity, noise, and cold-start
* ✅ Deployed a scalable machine learning system on cloud
* ✅ Developed full-stack data-driven web applications

---

## 🚀 Future Enhancements

* ✅ Add user login/session support
* ✅ Save and track watch history
* ✅ Add dynamic genre filters
* ✅ Use **Neural Collaborative Filtering**
* ✅ Integrate movie trailer previews via YouTube API
* ✅ Enable multilingual support

---

## 🙋‍♀️ Author

**Sanskriti Sourya**
🎓 B.Tech, Computer Science – MNIT Jaipur (2021–2025)
📧 [sanskritisourya8448@gmail.com](mailto:sanskritisourya8448@gmail.com)
📞 +91-8825222820
🌐 [GitHub](https://github.com/Sanskritisouryaagarwal) | [LinkedIn](https://linkedin.com/in/sanskriti-sourya)

---


## 🙌 Acknowledgments

* [MovieLens Dataset – GroupLens Research](https://grouplens.org/datasets/movielens/)
* [The Movie Database API (TMDB)](https://www.themoviedb.org/)
* [OMDb API](https://www.omdbapi.com/)
* scikit-learn documentation for ML utilities

```

