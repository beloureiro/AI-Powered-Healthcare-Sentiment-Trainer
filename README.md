# AI-Powered Healthcare Sentiment Trainer

## Overview
AI-Powered Healthcare Sentiment Trainer is an advanced platform designed for **fine-tuning** AI models such as BERT, RoBERTa, and VADER to accurately analyze patient feedback within the healthcare sector. This platform is particularly focused on optimizing these models for sentiment analysis tailored to healthcare-specific data.

## Fine-Tuning Features
- **BERT Fine-Tuning**: Utilizes a pre-trained BERT model that has been fine-tuned on healthcare datasets to improve the accuracy of sentiment analysis related to patient feedback.
- **RoBERTa Fine-Tuning**: RoBERTa, a robust variant of BERT, is further fine-tuned to specialize in identifying subtle sentiment variations in healthcare reviews.
- **Twitter RoBERTa Model**: Specifically designed for analyzing sentiments in Twitter data, leveraging the nuances of social media language.
- **Multilingual BERT Model**: Capable of handling texts in multiple languages, providing sentiment analysis across diverse linguistic contexts.
- **VADER Sentiment**: Though VADER is a rule-based model, it complements the fine-tuned models by providing quick insights for shorter, direct feedback.
- **Customizable Sentiment Models**: The platform allows for ongoing fine-tuning and updates to the models as new data becomes available, ensuring adaptability and improving performance over time.

## Performance Dashboard
The platform includes an intuitive Streamlit-based dashboard that allows users to:
1. Input patient feedback.
2. View sentiment analysis results using fine-tuned models.
3. Track and compare model performance in real-time through dynamic visualizations.
4. Evaluate and store sentiment outcomes for further analysis.

## Database Integration
The results from sentiment analysis are stored in a connected database, allowing for long-term tracking and performance comparison of models over time.

## GPU Utilization
The models are optimized to utilize GPU resources when available, significantly enhancing processing speed and efficiency during sentiment analysis.

## Setup Instructions
To set up the tool, clone the repository and install the necessary dependencies.

```bash
# Clone the repository
git clone https://github.com/beloureiro/AI-Powered-Healthcare-Sentiment-Trainer.git

# Navigate to the repository
cd AI-Powered-Healthcare-Sentiment-Trainer

# Install required dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run main.py
```

## Usage
Once set up, users can input patient feedback through the dashboard and receive real-time sentiment analysis using fine-tuned AI models. Results are visualized and stored in the database.

```bash
# Example command to run sentiment analysis
python main.py --model bert --input data/reviews.csv
```

## Contributing
Feel free to fork the repository and submit pull requests to improve the tool or add new features.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
