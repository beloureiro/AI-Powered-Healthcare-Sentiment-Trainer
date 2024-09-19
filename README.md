# Healthcare Sentiment Trainer

## Project Overview

The Healthcare Sentiment Trainer is a Streamlit-based web application designed to analyze, compare, and continuously improve sentiment analysis models applied to patient reviews in the healthcare sector. This tool aims to help healthcare providers better understand patient feedback and improve their services based on sentiment analysis.

## Features

1. **Multi-model Sentiment Analysis**: Utilizes BERT, VADER, and RoBERTa models for comprehensive sentiment analysis.
2. **Interactive User Interface**: Built with Streamlit for an intuitive and responsive user experience.
3. **Sentiment Adjustment**: Allows users to fine-tune sentiment scores and categories.
4. **Touchpoint Validation**: Enables users to validate and adjust the healthcare process touchpoint associated with each review.
5. **Model Training**: Incorporates user feedback to continuously improve model performance.
6. **Performance Dashboard**: Tracks and visualizes model performance over time.

## Installation

To run the Healthcare Sentiment Trainer locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/healthcare-sentiment-trainer.git
   cd healthcare-sentiment-trainer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the database:
   - Ensure you have PostgreSQL installed and running.
   - Create a new database for the project.
   - Set the following environment variables with your database credentials:
     - `PGHOST`
     - `PGDATABASE`
     - `PGUSER`
     - `PGPASSWORD`
     - `PGPORT`

## Usage Instructions

1. Start the Streamlit app:
   ```
   streamlit run main.py
   ```

2. Open your web browser and navigate to `http://localhost:5000` (or the port specified in the terminal output).

3. Enter patient feedback in the provided text area.

4. Click the "Analyze Sentiment" button to process the feedback using multiple models.

5. Review the sentiment analysis results in the table and graph.

6. If needed, adjust the sentiment using the provided buttons and scale.

7. Validate or change the healthcare process touchpoint using the dropdown menu.

8. Click the "Train" button to update the models with your corrections.

9. Explore the Model Performance Dashboard to track sentiment trends and touchpoint distribution over time.

## Contributing

We welcome contributions to the Healthcare Sentiment Trainer! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request with a detailed description of your changes.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
