import streamlit as st
import plotly.graph_objects as go

TOUCHPOINTS = [
    "Search and Evaluate Professional Score",
    "Schedule Appointment",
    "Make Payment Online",
    "Make Payment at Reception",
    "Check-in Online",
    "Check-in at Reception",
    "Access Platform for Online Consultation",
    "Attend Online Consultation",
    "Attend Offline Consultation",
    "Follow-up Procedures",
    "Leave Review and Feedback"
]

def get_sentiment_category(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

def display_instructions():
    st.markdown("""
    ## Instructions:
    1. Enter patient feedback in the text box below.
    2. Click "Analyze Sentiment" to process the feedback.
    3. View the results in the table and graph.
    4. Adjust the sentiment using the buttons and scale if needed.
    5. Validate or change the healthcare process touchpoint.
    6. Click "Train" to update the models with your corrections.
    """)

def create_sentiment_plot(df):
    print("Creating sentiment plot with data:")
    print(df.to_string())
    colors = {'Positive': '#00854d', 'Neutral': '#94960a', 'Negative': '#990000'}
    fig = go.Figure()
    for category in df['Sentiment Category'].unique():
        category_data = df[df['Sentiment Category'] == category]
        fig.add_trace(go.Bar(
            x=category_data['Touchpoint'],
            y=category_data['Sentiment Score'],
            name=category,  
            marker_color=colors[category]
        ))
    fig.update_layout(
        title='Sentiment Analysis Results',
        xaxis_title='Healthcare Process Touchpoint',
        yaxis_title='Sentiment Score',
        barmode='group'
    )
    return fig
