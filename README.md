# 🏀 NBA Stat Predictor

An interactive **NBA Player Points Predictor** built with **Streamlit**, **nba_api**, and **scikit-learn**.  
It uses recent game logs (assists & rebounds) to predict a player's next game points using a trained **Linear Regression model**.

---

## 🚀 Features

✅ **Player Search** – Type any NBA player's name and fetch their recent games  
✅ **Real Data** – Uses [nba_api](https://github.com/swar/nba_api) to get live NBA stats  
✅ **Machine Learning Model** – Predicts points using AST + REB averages  
✅ **Interactive Web App** – Runs in your browser with no installation needed  
✅ **Easy Deployment** – Deployable to [Streamlit Cloud](https://streamlit.io/cloud) in minutes  

---

## 📸 Demo

![NBA Stat Predictor Screenshot](https://via.placeholder.com/800x400.png?text=NBA+Stat+Predictor+Demo)

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io/) – Interactive web UI
- [nba_api](https://github.com/swar/nba_api) – NBA stats data source
- [pandas](https://pandas.pydata.org/) – Data manipulation
- [scikit-learn](https://scikit-learn.org/) – Linear Regression model
- [joblib](https://joblib.readthedocs.io/) – Model persistence

## 🏗 Project Structure

```plaintext
NBA-Stat-Predictor/
│
├── app.py              # Streamlit web app
├── nba_model.pkl       # Trained linear regression model
├── player_gamelogs.csv # (optional) cached data for faster loading
├── requirements.txt    # Project dependencies
└── README.md           # This file
```
---

## ⚡ How to Run Locally

#### 1. Clone the repository
   ```bash
   git clone https://github.com/your-username/NBA-Stat-Predictor.git
   cd NBA-Stat-Predictor
   ```
 #### 2. Create & activate virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```
#### 4. Run Streamlit app

```bash
streamlit run app.py
```
#### 5. Open the link shown in your terminal (usually http://localhost:8501).

