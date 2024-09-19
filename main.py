import streamlit as st
from streamlit import session_state as state
import pandas as pd
import plotly.express as px
from models import BERTModel, VADERModel, RoBERTaModel
from database.db_operations import Database
from utils import TOUCHPOINTS, get_sentiment_category, display_instructions, create_sentiment_plot

# Initialize models and database
bert_model = BERTModel(model_dir="HealthcareSentiment_BERT_v1")
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
if 'bert_category' not in state:
    state.bert_category = None
if 'vader_category' not in state:
    state.vader_category = None
if 'roberta_category' not in state:
    state.roberta_category = None

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

            # Store sentiment categories in session state
            state.bert_category = bert_category
            state.vader_category = vader_category
            state.roberta_category = roberta_category

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

# Treinamento e captura dos valores ajustados
with st.form(key='train_form'):
    train_button = st.form_submit_button("Train")
    if train_button:
        if state.sentiment_score is not None and state.selected_touchpoint:
            with st.spinner("Training models... Please wait."):

                # Ajuste para pegar os valores ajustados pelo usuário
                adjusted_score = state.sentiment_score
                adjusted_touchpoint = state.selected_touchpoint

                # Definir a categoria de sentimento com base no valor ajustado
                post_sentiment_category = get_sentiment_category(adjusted_score)

                # Utilize esses valores no treinamento para os três modelos
                bert_model.train(state.feedback, state.bert_category, adjusted_score, adjusted_touchpoint)
                vader_model.train(state.feedback, state.vader_category, adjusted_score, adjusted_touchpoint)
                roberta_model.train(state.feedback, state.roberta_category, adjusted_score, adjusted_touchpoint)

                # Inserir os dados de treinamento no banco de dados para os três modelos
                db.insert_sentiment_data(
                    state.feedback,
                    'BERT',
                    state.results.loc[0, 'Sentiment Score'],
                    adjusted_score,
                    state.results.loc[0, 'Touchpoint'],
                    adjusted_touchpoint,
                    state.results.loc[0, 'Sentiment Category'],
                    post_sentiment_category  # Usar a categoria ajustada
                )
                db.insert_sentiment_data(
                    state.feedback,
                    'VADER',
                    state.results.loc[1, 'Sentiment Score'],
                    adjusted_score,
                    state.results.loc[1, 'Touchpoint'],
                    adjusted_touchpoint,
                    state.results.loc[1, 'Sentiment Category'],
                    post_sentiment_category  # Usar a categoria ajustada
                )
                db.insert_sentiment_data(
                    state.feedback,
                    'RoBERTa',
                    state.results.loc[2, 'Sentiment Score'],
                    adjusted_score,
                    state.results.loc[2, 'Touchpoint'],
                    adjusted_touchpoint,
                    state.results.loc[2, 'Sentiment Category'],
                    post_sentiment_category  # Usar a categoria ajustada
                )

                # Exibir os resultados pós-treinamento
                post_training_results = pd.DataFrame({
                    'Model': ['BERT', 'VADER', 'RoBERTa'],
                    'Post-Training Sentiment Score': [adjusted_score, adjusted_score, adjusted_score],
                    'Post-Training Sentiment Category': [post_sentiment_category, post_sentiment_category, post_sentiment_category],
                    'Post-Training Touchpoint': [adjusted_touchpoint, adjusted_touchpoint, adjusted_touchpoint]
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
    recent_df = pd.DataFrame(recent_data, columns=['Text', 'Model', 'pre_sentiment_score', 'post_sentiment_score', 'pre_touchpoint', 'post_touchpoint', 'pre_sentiment_category', 'post_sentiment_category'])
    st.dataframe(recent_df)
else:
    st.info("No recent sentiment data available.")

# Create dashboard for tracking model performance over time
st.header("Model Performance Dashboard")

# Fetch historical data from the database
historical_data = db.get_historical_sentiment_data()

if historical_data:
    df = pd.DataFrame(historical_data, columns=['created_at', 'model_name', 'pre_sentiment_score', 'post_sentiment_score', 'pre_touchpoint', 'post_touchpoint'])
    df['Date'] = pd.to_datetime(df['created_at'])
    df['Touchpoint'] = df['pre_touchpoint']  # or df['post_touchpoint'], depending on what you need
    df['Sentiment Category'] = df['pre_sentiment_score'].apply(get_sentiment_category)

    # Create a line plot for sentiment scores over time
    fig_sentiment = px.line(df, x='Date', y='pre_sentiment_score', title='Sentiment Scores Over Time')
    st.plotly_chart(fig_sentiment)

else:
    st.info("No historical data available for the dashboard. Please analyze and train on some feedback to populate the dashboard.")

# Close the database connection when the app is done
db.close()
