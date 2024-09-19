import streamlit as st
from streamlit import session_state as state
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from models import BERTModel, VADERModel, RoBERTaModel
from database.db_operations import Database
from utils import TOUCHPOINTS, get_sentiment_category, display_instructions, create_sentiment_plot

# Initialize models and database
bert_model = BERTModel()
vader_model = VADERModel()
roberta_model = RoBERTaModel()
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

# Set page config
st.set_page_config(page_title="Healthcare Sentiment Trainer", page_icon="🏥", layout="wide")

# Display logo
st.image("Logo/Design_InMotion_Transp_wilde.png", width=600)

# Display title
st.title("Healthcare Sentiment Trainer")

# Display instructions
display_instructions()

# Input text box for patient feedback
state.feedback = st.text_area("Enter patient feedback here:", value=state.feedback, height=100)

# Analyze sentiment form
with st.form(key='analyze_form'):
    analyze_button = st.form_submit_button("Analyze Sentiment")
    if analyze_button and state.feedback:
        with st.spinner("Analyzing feedback... Please wait."):
            # Process feedback with all models
            bert_score, bert_category, bert_touchpoint = bert_model.analyze(state.feedback)
            vader_score, vader_category, vader_touchpoint = vader_model.analyze(state.feedback)
            roberta_score, roberta_category, roberta_touchpoint = roberta_model.analyze(state.feedback)

            # Create DataFrame with results
            state.results = pd.DataFrame({
                'Model': ['BERT', 'VADER', 'RoBERTa'],
                'Sentiment Score': [bert_score, vader_score, roberta_score],
                'Sentiment Category': [bert_category, vader_category, roberta_category],
                'Touchpoint': [bert_touchpoint, vader_touchpoint, roberta_touchpoint]
            })

# Sentiment adjustment form
with st.form(key='sentiment_adjustment_form'):
    st.subheader("Adjust Sentiment")
    col1, col2, col3 = st.columns(3)
    with col1:
        negative = st.form_submit_button("Negative")
    with col2:
        neutral = st.form_submit_button("Neutral")
    with col3:
        positive = st.form_submit_button("Positive")
    
    if negative:
        state.sentiment_score = st.slider("Fine-tune negative sentiment", -1.0, 0.0, -0.5, 0.01, key='neg_slider')
    elif neutral:
        state.sentiment_score = 0.0
        st.info("Neutral sentiment set to 0.0")
    elif positive:
        state.sentiment_score = st.slider("Fine-tune positive sentiment", 0.0, 1.0, 0.5, 0.01, key='pos_slider')
    
    # Touchpoint validation
    state.selected_touchpoint = st.selectbox("Validate or change the healthcare process touchpoint:", TOUCHPOINTS, index=TOUCHPOINTS.index(state.selected_touchpoint) if state.selected_touchpoint else 0)

# Train button logic inside a form
with st.form(key='train_form'):
    train_button = st.form_submit_button("Train")
    if train_button:
        if state.sentiment_score is not None and state.selected_touchpoint:
            with st.spinner("Training models... Please wait."):
                
                # Capture the pre-training results
                pre_bert_score, pre_bert_category, pre_bert_touchpoint = bert_model.analyze(state.feedback)
                pre_vader_score, pre_vader_category, pre_vader_touchpoint = vader_model.analyze(state.feedback)
                pre_roberta_score, pre_roberta_category, pre_roberta_touchpoint = roberta_model.analyze(state.feedback)

                # Display the pre-training results (optional)
                pre_training_results = pd.DataFrame({
                    'Model': ['BERT', 'VADER', 'RoBERTa'],
                    'Pre-Training Sentiment Score': [pre_bert_score, pre_vader_score, pre_roberta_score],
                    'Pre-Training Sentiment Category': [pre_bert_category, pre_vader_category, pre_roberta_category],
                    'Pre-Training Touchpoint': [pre_bert_touchpoint, pre_vader_touchpoint, pre_roberta_touchpoint]
                })
                st.dataframe(pre_training_results)

                # Update the models (training step)
                bert_model.train(state.feedback, state.sentiment_score, state.selected_touchpoint)
                vader_model.train(state.feedback, state.sentiment_score, state.selected_touchpoint)
                roberta_model.train(state.feedback, state.sentiment_score, state.selected_touchpoint)

                # Capture the post-training results
                post_bert_score, post_bert_category, post_bert_touchpoint = bert_model.analyze(state.feedback)
                post_vader_score, post_vader_category, post_vader_touchpoint = vader_model.analyze(state.feedback)
                post_roberta_score, post_roberta_category, post_roberta_touchpoint = roberta_model.analyze(state.feedback)

                # Insert both pre- and post-training data into the database
                db.insert_sentiment_data(
                    state.feedback,
                    pre_bert_score, post_bert_score, pre_bert_touchpoint, post_bert_touchpoint
                )

                # Display the post-training results
                post_training_results = pd.DataFrame({
                    'Model': ['BERT', 'VADER', 'RoBERTa'],
                    'Post-Training Sentiment Score': [post_bert_score, post_vader_score, post_roberta_score],
                    'Post-Training Sentiment Category': [post_bert_category, post_vader_category, post_roberta_category],
                    'Post-Training Touchpoint': [post_bert_touchpoint, post_vader_touchpoint, post_roberta_touchpoint]
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
    # Create DataFrame with all returned columns
    recent_df = pd.DataFrame(recent_data, columns=['Text', 'Pre-Sentiment Score', 'Post-Sentiment Score', 'Pre-Touchpoint', 'Post-Touchpoint'])
    st.dataframe(recent_df)
else:
    st.info("No recent sentiment data available.")


# Create dashboard for tracking model performance over time
st.header("Model Performance Dashboard")

# Fetch historical data from the database
historical_data = db.get_historical_sentiment_data()

if historical_data:
    df = pd.DataFrame(historical_data, columns=['Date', 'Pre-Sentiment Score', 'Post-Sentiment Score', 'Pre-Touchpoint', 'Post-Touchpoint'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Touchpoint'] = df['Pre-Touchpoint']  # or df['Post-Touchpoint'], depending on what you need
    df['Sentiment Category'] = df['Pre-Sentiment Score'].apply(get_sentiment_category)

    # Create a line plot for sentiment scores over time
    fig_sentiment = px.line(df, x='Date', y='Pre-Sentiment Score', title='Sentiment Scores Over Time')
    st.plotly_chart(fig_sentiment)

    # Create a bar chart for touchpoint distribution
    fig_touchpoints = px.bar(df['Touchpoint'].value_counts().reset_index(), x='Touchpoint', y='count', title='Touchpoint Distribution')
    fig_touchpoints.update_layout(xaxis_title='Touchpoint', yaxis_title='Count')
    st.plotly_chart(fig_touchpoints)

    # Create a pie chart for sentiment category distribution
    fig_sentiment_dist = px.pie(df, names='Sentiment Category', title='Sentiment Category Distribution')
    st.plotly_chart(fig_sentiment_dist)

    # Display some summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Analyzed Feedbacks", len(df))
    with col2:
        st.metric("Average Sentiment Score", f"{df['Pre-Sentiment Score'].mean():.2f}")
    with col3:
        st.metric("Most Common Touchpoint", df['Touchpoint'].mode().values[0])

else:
    st.info("No historical data available for the dashboard. Please analyze and train on some feedback to populate the dashboard.")

# Close the database connection when the app is done
db.close()
