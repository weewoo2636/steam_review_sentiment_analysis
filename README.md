
# Steam Review Sentiment Analysis

This project was made to help a fictional game development studio to get a better grasp of how their game is perceived on the Steam platform, by creating a steam review sentiment analysis, that could help the studio make a better decision. The model got 0.81 f1-score.


## Background

With the vast amount of data generated from user reviews on Steam, both developers and players benefit from systems that can aggregate and summarize the sentiment expressed. This project is motivated by several key factors:

Game Developers: Developers can use sentiment analysis to better understand player feedback, helping with game updates, bug fixes, or future releases.
Players: Players can get a quick, high-level overview of how others feel about a game before making a purchase decision.
Market Analysis: Sentiment trends can be correlated with game sales, updates, or marketing efforts, providing valuable insights into market dynamics.


## Acknowledgements

- [Dataset](https://www.kaggle.com/datasets/arashnic/game-review-dataset)


## Methods & Tech Stack

- **Data Pre-Processing & Exploration:** Python, Pandas, Numpy, Matplotlib, Seaborn, Wordcloud

- **Machine Learning:** Tensorflow Keras, Nltk, Sklearn, Pickle

- **Deployment:** Streamlit


## Workflow

Data Loading -> EDA -> Feature Engineering -> Model Training -> Model Saving -> Model Inference Testing -> Model Deployment


## Screenshots

![Deployment Screenshot](https://github.com/weewoo2636/steam_review_sentiment_analysis/blob/main/screenshot.png?raw=true)


## Output/Deployment

To see the deployed model, visit here: [link_to_deployed_model](https://huggingface.co/spaces/weewoo2636/P2G7_wilson_deployment)


## Files Overview

1. **`main_notebook.ipynb`**: Primary Jupyter Notebook containing the full pipeline for data preprocessing, model training, and sentiment analysis of Steam reviews.
   
2. **`inference_notebook.ipynb`**: Notebook for loading the pre-trained model and making predictions on new Steam review data.

3. **`train.csv`**: The dataset used for training the sentiment analysis model, containing raw review data.

4. **`train_trimmed.csv`**: A cleaned or smaller version of the training dataset to facilitate quicker model training.

5. **`model.h5`**: Saved Keras model file, used for sentiment prediction during inference or deployment.

6. **`vectorizer.pkl`**: A Pickle file that stores the vectorizer (e.g., `TfidfVectorizer`) for transforming review text into numerical form for model input.

7. **`url.txt`**: Contains the deployment URL, possibly a link to the model hosted on platforms like Hugging Face Spaces.

8. **`eda_1.png`, `eda_2.png`, `eda_3.png`**: Images displaying results from exploratory data analysis (EDA).

9. **`model.png`**: A visual representation of the machine learning model's architecture.

10. **`README.md`**: Project documentation providing an overview of the project, instructions for setup, and performance details.

### `/deployment` Folder

11. **`app.py`**: The main application file (likely for Streamlit) that contains code for the web interface where users can input reviews and get sentiment predictions from the trained model.

12. **`requirements.txt`**: Lists the Python dependencies and packages required to run the application and model. It ensures the deployment environment installs the correct libraries.

13. **`Procfile`**: A file likely used for deploying the app on platforms like Heroku. It specifies the commands to start the web application.


## Authors

- [@weewoo2636](https://www.github.com/weewoo2636)

