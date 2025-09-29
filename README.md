# Real-Time-news-Sentiment-Dashboard
🚀 Real-time news sentiment analysis dashboard using PySpark for the ML pipeline and Streamlit for interactive visualization. Fetches live data from NewsAPI.

# Real-Time News Sentiment Analysis with PySpark and Streamlit

A complete data pipeline that fetches live news headlines, analyzes their sentiment using a PySpark ML model, and displays the results on an interactive Streamlit dashboard.

<img width="1920" height="1080" alt="Screenshot 2025-09-28 212635" src="https://github.com/user-attachments/assets/3dedcdf9-902e-46fb-a5a9-3eb0dc45f975" />
<img width="1920" height="1080" alt="Screenshot 2025-09-28 212658" src="https://github.com/user-attachments/assets/4b76494f-d1ed-4a5b-8783-37750a8c28a5" />
<img width="1920" height="928" alt="image" src="https://github.com/user-attachments/assets/c0ad6f41-6bbf-4475-ac76-019c1e3d6e0a" />



## Features
- Fetches real-time news articles from NewsAPI.
- Preprocesses text data and trains a Logistic Regression model using a PySpark ML Pipeline.
- Achieves ~80% accuracy on sentiment classification (positive/negative).
- Deploys a user-friendly Streamlit dashboard to visualize sentiment distribution in real-time.

## Tech Stack
- **Data Processing & ML:** PySpark
- **Web Dashboard:** Streamlit, Plotly
- **Data Source:** NewsAPI
- **Deployment:** ngrok (for local tunneling)

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up your API Key:**
    - Get a free API key from [newsapi.org](https://newsapi.org).
    - Create a file `.streamlit/secrets.toml` and add your key: `NEWS_API_KEY = "YOUR_KEY_HERE"`
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
