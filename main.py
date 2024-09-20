import streamlit as st
from streamlit import session_state as state
import pandas as pd
import plotly.express as px
from models import (
    BERTModel,
    VADERModel,
    RoBERTaModel,
    MultilingualBERTModel,
    TwitterRoBERTaModel,
    # BERTweetModel,  # Remover esta linha
)
from database.db_operations import Database
from utils import TOUCHPOINTS, get_sentiment_category, display_instructions, create_sentiment_plot

# Initialize models and database
bert_model = BERTModel(model_dir="HealthcareSentiment_BERT_v1")
vader_model = VADERModel()
roberta_model = RoBERTaModel()
multilingual_bert_model = MultilingualBERTModel()
twitter_roberta_model = TwitterRoBERTaModel()
# bertweet_model = BERTweetModel()  # Remover esta linha
db = Database()

# Initialize session state variables
if 'feedback' not in state:
    state.feedback = ""
if 'sentiment_score' not in state:
    state.sentiment_score = None
if 'selected_touchpoint' not in state:
    state.selected_touchpoint = None
if 'results' not in state:
    state.results = None
if 'bert_category' not in state:
    state.bert_category = None
if 'vader_category' not in state:
    state.vader_category = None
if 'roberta_category' not in state:
    state.roberta_category = None
if 'multilingual_category' not in state:
    state.multilingual_category = None
if 'twitter_category' not in state:
    state.twitter_category = None
# if 'bertweet_category' not in state:
#     state.bertweet_category = None  # Remover esta linha

# Set page config
st.set_page_config(page_title="AI-Powered Sentiment Trainer", page_icon="🏥", layout="wide")

# Display logo
st.image("Logo/Design_InMotion_Transp_wilde.png", width=600)

# Display title
st.title(":green[_AI-Powered_] Healthcare Sentiment :green[Trainer]")

# Display instructions
display_instructions()

# Input text box for patient feedback
state.feedback = st.text_area(
    "",
    value=state.feedback,
    height=150,
    placeholder="Enter patient feedback here..."
)

# Function to map MultilingualBERTModel categories to standard sentiment categories
def map_multilingual_category(category):
    if category in ['1 star', '2 stars']:
        return 'Negative'
    elif category == '3 stars':
        return 'Neutral'
    elif category in ['4 stars', '5 stars']:
        return 'Positive'
    else:
        return 'Neutral'  # default

# Analyze sentiment form
with st.form(key='analyze_form'):
    analyze_button = st.form_submit_button("Analyze Sentiment")
    if analyze_button and state.feedback:
        with st.spinner("Analyzing feedback... Please wait."):
            # Process feedback with all models
            bert_score, bert_category, bert_touchpoint = bert_model.analyze(state.feedback)
            vader_score, vader_category, vader_touchpoint = vader_model.analyze(state.feedback)
            roberta_score, roberta_category, roberta_touchpoint = roberta_model.analyze(state.feedback)
            multilingual_score, multilingual_category, multilingual_touchpoint = multilingual_bert_model.analyze(state.feedback)
            twitter_score, twitter_category, twitter_touchpoint = twitter_roberta_model.analyze(state.feedback)
            # bertweet_score, bertweet_category, bertweet_touchpoint = bertweet_model.analyze(state.feedback)  # Remover esta linha

            # Store sentiment categories in session state
            state.bert_category = bert_category
            state.vader_category = vader_category
            state.roberta_category = roberta_category
            state.multilingual_category = multilingual_category
            state.twitter_category = twitter_category
            # state.bertweet_category = bertweet_category  # Remover esta linha

            # Create DataFrame with results
            state.results = pd.DataFrame({
                'Model': ['BERT', 'VADER', 'RoBERTa', 'MultilingualBERT', 'TwitterRoBERTa'],  # Remover 'BERTweet'
                'Sentiment Score': [
                    bert_score,
                    vader_score,
                    roberta_score,
                    multilingual_score,
                    twitter_score,
                    # bertweet_score  # Remover esta linha
                ],
                'Sentiment Category': [
                    bert_category,
                    vader_category,
                    roberta_category,
                    multilingual_category,
                    twitter_category,
                    # bertweet_category  # Remover esta linha
                ],
                'Touchpoint': [
                    bert_touchpoint,
                    vader_touchpoint,
                    roberta_touchpoint,
                    multilingual_touchpoint,
                    twitter_touchpoint,
                    # bertweet_touchpoint  # Remover esta linha
                ]
            })

# Sentiment adjustment form
with st.form(key='sentiment_adjustment_form'):
    st.subheader("_Adjust Sentiment_", divider="green")
    col1, col2, col3 = st.columns(3)
    with col1:
        negative = st.form_submit_button("Negative")
    with col2:
        neutral = st.form_submit_button("Neutral")
    with col3:
        positive = st.form_submit_button("Positive")

    if negative:
        state.sentiment_score = st.slider(
            "Fine-tune negative sentiment",
            -1.0,
            0.0,
            -0.5,
            0.01,
            key='neg_slider'
        )
    elif neutral:
        state.sentiment_score = 0.0
        st.info("Neutral sentiment set to 0.0")
    elif positive:
        state.sentiment_score = st.slider(
            "Fine-tune positive sentiment",
            0.0,
            1.0,
            0.5,
            0.01,
            key='pos_slider'
        )

    # Touchpoint validation
    state.selected_touchpoint = st.selectbox(
        "Validate or change the healthcare process touchpoint:",
        TOUCHPOINTS,
        index=TOUCHPOINTS.index(state.selected_touchpoint) if state.selected_touchpoint else 0
    )

    # Move the "Train" button to be below the touchpoint selector
    train_button = st.form_submit_button("Train")
    if train_button:
        if state.sentiment_score is not None and state.selected_touchpoint:
            with st.spinner("Training models... Please wait."):
                # Adjust to get the values adjusted by the user
                adjusted_score = state.sentiment_score
                adjusted_touchpoint = state.selected_touchpoint

                # Define the sentiment category based on the adjusted value
                post_sentiment_category = get_sentiment_category(adjusted_score)

                # Use these values in training for the three main models
                bert_model.train(state.feedback, adjusted_score, adjusted_touchpoint)
                vader_model.train(state.feedback, adjusted_score, adjusted_touchpoint)
                roberta_model.train(state.feedback, adjusted_score, adjusted_touchpoint)

                # Insert training data into the database for the three models
                db.insert_sentiment_data(
                    state.feedback,
                    'BERT',
                    state.results.loc[0, 'Sentiment Score'],
                    adjusted_score,
                    state.results.loc[0, 'Touchpoint'],
                    adjusted_touchpoint,
                    state.results.loc[0, 'Sentiment Category'],
                    post_sentiment_category  # Use adjusted category
                )
                db.insert_sentiment_data(
                    state.feedback,
                    'VADER',
                    state.results.loc[1, 'Sentiment Score'],
                    adjusted_score,
                    state.results.loc[1, 'Touchpoint'],
                    adjusted_touchpoint,
                    state.results.loc[1, 'Sentiment Category'],
                    post_sentiment_category  # Use adjusted category
                )
                db.insert_sentiment_data(
                    state.feedback,
                    'RoBERTa',
                    state.results.loc[2, 'Sentiment Score'],
                    adjusted_score,
                    state.results.loc[2, 'Touchpoint'],
                    adjusted_touchpoint,
                    state.results.loc[2, 'Sentiment Category'],
                    post_sentiment_category  # Use adjusted category
                )
                db.insert_sentiment_data(
                    state.feedback,
                    'MultilingualBERT',
                    state.results.loc[3, 'Sentiment Score'],
                    adjusted_score,
                    state.results.loc[3, 'Touchpoint'],
                    adjusted_touchpoint,
                    state.results.loc[3, 'Sentiment Category'],
                    post_sentiment_category  # Use adjusted category
                )
                db.insert_sentiment_data(
                    state.feedback,
                    'TwitterRoBERTa',
                    state.results.loc[4, 'Sentiment Score'],
                    adjusted_score,
                    state.results.loc[4, 'Touchpoint'],
                    adjusted_touchpoint,
                    state.results.loc[4, 'Sentiment Category'],
                    post_sentiment_category  # Use adjusted category
                )

                # Display post-training results
                post_training_results = pd.DataFrame({
                    'Model': ['BERT', 'VADER', 'RoBERTa'],
                    'Post-Training Sentiment Score': [adjusted_score, adjusted_score, adjusted_score],
                    'Post-Training Sentiment Category': [
                        post_sentiment_category,
                        post_sentiment_category,
                        post_sentiment_category
                    ],
                    'Post-Training Touchpoint': [
                        adjusted_touchpoint,
                        adjusted_touchpoint,
                        adjusted_touchpoint
                    ]
                })
                st.dataframe(post_training_results)

                st.success("Models trained and data saved successfully!")

# Display current results
if state.results is not None:
    st.subheader("Current Sentiment Analysis Results")
    st.dataframe(state.results)
    current_fig = create_sentiment_plot(state.results)
    st.plotly_chart(current_fig)

# Display recent sentiment data
st.subheader("Recent Sentiment Data")
recent_data = db.get_recent_sentiment_data()
if recent_data:
    # Create DataFrame with all returned columns, including 'model_name'
    recent_df = pd.DataFrame(
        recent_data,
        columns=[
            'Text',
            'Model',
            'pre_sentiment_score',
            'post_sentiment_score',
            'pre_touchpoint',
            'post_touchpoint',
            'pre_sentiment_category',
            'post_sentiment_category'
        ]
    )
    st.dataframe(recent_df)
else:
    st.info("No recent sentiment data available.")

# Create dashboard for tracking model performance over time
st.header("Model Performance Dashboard")

# Fetch historical data from the database
historical_data = db.get_historical_sentiment_data()

if historical_data:
    df = pd.DataFrame(
        historical_data,
        columns=[
            'created_at',
            'model_name',
            'pre_sentiment_score',
            'post_sentiment_score',
            'pre_touchpoint',
            'post_touchpoint'
        ]
    )
    df['Date'] = pd.to_datetime(df['created_at'])
    df['Touchpoint'] = df['pre_touchpoint']  # or df['post_touchpoint', depending on what you need]
    df['Sentiment Category'] = df['pre_sentiment_score'].apply(get_sentiment_category)

    # Create a line plot for sentiment scores over time
    fig_sentiment = px.line(
        df,
        x='Date',
        y='pre_sentiment_score',
        color='model_name',
        title='Sentiment Scores Over Time'
    )
    st.plotly_chart(fig_sentiment)

else:
    st.info("No historical data available for the dashboard. Please analyze and train on some feedback to populate the dashboard.")

# Close the database connection when the app is done
db.close()
