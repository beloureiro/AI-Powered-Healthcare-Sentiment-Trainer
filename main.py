import streamlit as st
from streamlit import session_state as state
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from models import BERTModel, VADERModel, RoBERTaModel
from database.db_operations import Database
from utils import TOUCHPOINTS, get_sentiment_category, display_instructions, create_sentiment_plot, suggest_touchpoint

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
st.image("Logo/Design_InMotion_Transp.png", width=700)

# Display title
st.title("Healthcare Sentiment Trainer")

# Display instructionsP
display_instructions()

# Input text box for patient feedback
state.feedback = st.text_area("Enter patient feedback here:", value=state.feedback, height=100)

# Analyze sentiment form
with st.form(key='analyze_form'):
    analyze_button = st.form_submit_button("Analyze Sentiment")
    if analyze_button and state.feedback:
        with st.spinner("Analyzing feedback... Please wait."):
            # Process feedback with all models
            bert_score, bert_category = bert_model.predict(state.feedback)
            vader_score, vader_category = vader_model.predict(state.feedback)
            roberta_score, roberta_category = roberta_model.predict(state.feedback)

            # Suggest touchpoint
            suggested_touchpoint = suggest_touchpoint(state.feedback)

            # Create DataFrame with results
            state.results = pd.DataFrame({
                'Model': ['BERT', 'VADER', 'RoBERTa'],
                'Sentiment Score': [bert_score, vader_score, roberta_score],
                'Sentiment Category': [bert_category, vader_category, roberta_category],
                'Touchpoint': [suggested_touchpoint] * 3
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

    # Train button
    if st.form_submit_button("Train"):
        if state.sentiment_score is not None and state.selected_touchpoint:
            with st.spinner("Training models... Please wait."):
                # Update the models
                bert_model.train(state.feedback, state.sentiment_score)
                vader_model.train(state.feedback, state.sentiment_score)
                roberta_model.train(state.feedback, state.sentiment_score)

                # Store the data in the database
                db.insert_sentiment_data(state.feedback, state.sentiment_score, state.selected_touchpoint)

                # Update results with new predictions
                bert_score, bert_category = bert_model.predict(state.feedback)
                vader_score, vader_category = vader_model.predict(state.feedback)
                roberta_score, roberta_category = roberta_model.predict(state.feedback)

                new_results = pd.DataFrame({
                    'Model': ['BERT', 'VADER', 'RoBERTa'],
                    'Sentiment Score': [bert_score, vader_score, roberta_score],
                    'Sentiment Category': [bert_category, vader_category, roberta_category],
                    'Touchpoint': [state.selected_touchpoint] * 3
                })

                # Update state.results
                state.results = new_results

                st.success("Models updated and data stored successfully!")

                # Force a rerun of the app
                st.rerun()

        else:
            st.warning("Please adjust the sentiment and select a touchpoint before training.")

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
    recent_df = pd.DataFrame(recent_data, columns=['Text', 'Sentiment Score', 'Touchpoint'])
    st.dataframe(recent_df)
else:
    st.info("No recent sentiment data available.")

# Create dashboard for tracking model performance over time
st.header("Model Performance Dashboard")

# Fetch historical data from the database
historical_data = db.get_historical_sentiment_data()

if historical_data:
    df = pd.DataFrame(historical_data, columns=['Date', 'Sentiment Score', 'Touchpoint'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Sentiment Category'] = df['Sentiment Score'].apply(get_sentiment_category)

    # Create a line plot for sentiment scores over time
    fig_sentiment = px.line(df, x='Date', y='Sentiment Score', title='Sentiment Scores Over Time')
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
        st.metric("Average Sentiment Score", f"{df['Sentiment Score'].mean():.2f}")
    with col3:
        st.metric("Most Common Touchpoint", df['Touchpoint'].mode().values[0])

else:
    st.info("No historical data available for the dashboard. Please analyze and train on some feedback to populate the dashboard.")

# Close the database connection when the app is done
db.close()
