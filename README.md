# Sentiment Analysis with RNN
This project demonstrates how to perform Sentiment Analysis using a Recurrent Neural Network (RNN) model built with Python. The model is trained on labeled text data to predict whether a given text expresses a positive or negative sentiment.

# Table of Contents
About
Technologies Used
How to Run
Results
Future Work
License

# About
Sentiment Analysis is a common Natural Language Processing (NLP) task where the goal is to classify the sentiment behind a text (positive, negative, or neutral).
This project uses an RNN architecture to capture the sequential nature of text data and predict sentiment labels.

# Key steps include:
Preprocessing the dataset (tokenization, padding, etc.)
Building the RNN model
Training and evaluating the model
Visualizing the performance

# Technologies Used
Python 3.x
TensorFlow / Keras
NumPy
Matplotlib
Scikit-learn (for metrics like confusion matrix)

# How to Run
Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/sentiment-analysis-rnn.git
cd sentiment-analysis-rnn

Install the required packages:
bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook sentiment_analysis_with_rnn.ipynb
Make sure you have Jupyter installed (pip install notebook).

# Results
The RNN model was trained successfully and achieved good performance on the test dataset.
You can find detailed training curves (accuracy, loss) and confusion matrix plots inside the notebook.

Sample prediction:
Text	Predicted Sentiment
"This movie was amazing!"	Positive
"I did not enjoy the film."	Negative

# Future Work
Implement LSTM or GRU layers to improve performance.
Try using pre-trained embeddings like GloVe or Word2Vec.
Experiment with bidirectional RNNs
Fine-tune hyperparameters for better accuracy.

# License
This project is open source and available under the MIT License.

# Sentiment Analysis with an Recurrent Neural Networks (RNN)
Recurrent Neural Networks (RNNs) excel in sequence tasks such as sentiment analysis due to their ability to capture context from sequential data. In this article we will be apply RNNs to analyze the sentiment of customer reviews from Swiggy food delivery platform. The goal is to classify reviews as positive or negative for providing insights into customer experiences.

We will conduct a Sentiment Analysis using the TensorFlow framework:

# Step 1: Importing Libraries and Dataset
# Step 2: Loading Dataset
# Step 3: Text Cleaning and Sentiment Labeling
data[‘sentiment’]: Uses Avg Rating to generate binary labels (positive if rating >3.5)
# Tokenization and Padding
Tokenizer: Converts words into integer sequences.
Padding: Ensures all input sequences have the same length (max_length).
Note: These concepts are a not a part of RNN but are done to make model prediction better. You can refer to tokenization and padding for more details.
# Data Splitting
# Build RNN Model
Embedding Layer: Converts integer sequences into dense vectors (16 dimensions).
RNN Layer: Processes sequence data with 64 units and tanh activation.
Output Layer: Predicts sentiment probability using sigmoid activation.
Train & Evaluate Model
Epochs: Number of training iterations i.e 5
Batch Size: Processes 32 samples per gradient update.
Our model achieved a accuracy of 72% which is great for a RNN model. We can further fine tune it to achieve more accuracy.

# Predicting Sentiment
In summary the model processes textual reviews through RNN to predict sentiment from raw data. This helps in actionable insights by understanding customer sentiment.

