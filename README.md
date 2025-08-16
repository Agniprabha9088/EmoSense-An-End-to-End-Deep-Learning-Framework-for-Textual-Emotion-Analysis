# Introducing EmoSense: An End-to-End Deep Learning Framework for Textual Emotion Analysis

[](https://www.python.org/downloads/)
[](https://www.tensorflow.org/)
[](https://opensource.org/licenses/MIT)

A deep learning project for multi-class emotion classification of text data. This model leverages a Bidirectional Long Short-Term Memory (LSTM) network built with TensorFlow and Keras to classify tweets into one of six distinct emotions.

-----

## 📊 Project Overview

This project addresses the task of detecting emotions from written text, a key challenge in Natural Language Processing (NLP). By analyzing textual data, the model can infer the underlying emotional tone, which has applications in sentiment analysis, customer feedback monitoring, mental health tracking, and content recommendation systems.

The solution uses a sequential model featuring an Embedding layer and two stacked Bidirectional LSTM layers, which allows the network to capture contextual information from both forward and backward directions in a sentence, leading to a more nuanced understanding of the text.

The model is trained on the popular **`dair-ai/emotion`** dataset, which contains thousands of labeled tweets.

*Training history showing the model's learning progression over 20 epochs.*

-----

## ✨ Features

  * **Multi-Class Classification**: Classifies text into 6 emotions: **sadness**, **joy**, **love**, **anger**, **fear**, and **surprise**.
  * **Deep Learning Architecture**: Utilizes a powerful Bidirectional LSTM network to understand long-range dependencies and context in text.
  * **Data Preprocessing**: Implements a standard NLP pipeline including tokenization, padding, and label encoding.
  * **Performance Visualization**: Includes helper functions to generate plots for training/validation accuracy and loss, as well as a confusion matrix to evaluate model performance on the test set.

-----

## 🧠 Model Architecture

The model is a `tf.keras.models.Sequential` stack, designed to process sequences of text effectively.

1.  **Embedding Layer**:

      * `tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=60)`
      * This layer converts integer-encoded text (word indices) into dense vectors of a fixed size (16 dimensions). It learns a meaningful vector representation for each of the 10,000 words in the vocabulary.

2.  **First Bidirectional LSTM Layer**:

      * `tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True))`
      * Processes the sequence of word embeddings. The `Bidirectional` wrapper allows the LSTM to learn from both past and future context. `return_sequences=True` is crucial as it ensures the layer outputs a full sequence to be fed into the next LSTM layer.

3.  **Second Bidirectional LSTM Layer**:

      * `tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20))`
      * This layer processes the output from the previous Bi-LSTM layer for further feature extraction. It outputs only the final hidden state, which summarizes the entire sequence.

4.  **Output Layer**:

      * `tf.keras.layers.Dense(6, activation='softmax')`
      * A fully connected layer with 6 neurons (one for each emotion class). The `softmax` activation function converts the logits into a probability distribution, indicating the model's confidence for each emotion.

The model is compiled using the `adam` optimizer and `sparse_categorical_crossentropy` loss function, suitable for multi-class integer-labeled classification problems.

-----

## 💾 Dataset

The model is trained and evaluated on the **dair-ai/emotion** dataset, available on the Hugging Face Hub.

  * **Source**: [https://huggingface.co/datasets/dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
  * **Description**: This dataset contains English Twitter messages labeled with one of six basic emotions.
  * **Splits**:
      * **Training**: 16,000 samples
      * **Validation**: 2,000 samples
      * **Test**: 2,000 samples

-----

## ⚙️ Technologies Used

  * **Framework**: TensorFlow, Keras
  * **Data Manipulation**: Pandas, NumPy
  * **Data Visualization**: Matplotlib
  * **NLP & Evaluation**: Scikit-learn, Datasets (`nlp` library)
  * **Environment**: Jupyter Notebook

-----

## 🚀 Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Emotion-Detection-using-LSTM.git
    cd Emotion-Detection-using-LSTM
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file should be created with the following content:

    ```
    tensorflow
    pandas
    numpy
    matplotlib
    scikit-learn
    datasets
    pyarrow
    ```

    Then, run the installation command:

    ```bash
    pip install -r requirements.txt
    ```

-----

## USAGE

1.  Launch Jupyter Notebook or Jupyter Lab:
    ```bash
    jupyter notebook
    ```
2.  Open the `.ipynb` file containing the project code.
3.  Run the cells sequentially to load the data, build the model, train it, and evaluate its performance.

-----

## 📈 Results & Performance

The model achieves strong performance on the classification task.

  * **Training Accuracy**: **\~99.18%**
  * **Validation Accuracy**: **\~89.05%**

The significant gap between training and validation accuracy suggests that the model is **overfitting**. While it has learned the training data very well, its ability to generalize to unseen data is limited. The confusion matrix below provides a detailed breakdown of its performance on the test set.

*Confusion matrix showing the model's predictions vs. the true labels for the test dataset.*

-----

## 💡 Future Improvements

To improve generalization and combat overfitting, several strategies could be explored:

1.  **Regularization**: Introduce Dropout layers between the LSTM and Dense layers to prevent co-adaptation of neurons. L1/L2 regularization could also be added to the layers.
2.  **Use Pre-trained Embeddings**: Instead of learning embeddings from scratch, leverage pre-trained word embeddings like GloVe, Word2Vec, or FastText. This can provide a better starting point and improve performance, especially with limited data.
3.  **Advanced Architectures**: Experiment with more complex models like GRUs (which are computationally less expensive than LSTMs) or Transformer-based models (e.g., using a fine-tuned BERT or DistilBERT) for potentially higher accuracy.
4.  **Hyperparameter Tuning**: Systematically tune hyperparameters like the embedding dimension, LSTM units, learning rate, and batch size using techniques like KerasTuner or Grid Search.
5.  **Deployment**: Wrap the trained model in a REST API using Flask or FastAPI to create a web service for real-time emotion prediction.

-----


## 🙏 Acknowledgements

  * This project uses the `emotion` dataset provided by `dair-ai` on Hugging Face.
  * Built with the power of the open-source community, especially the teams behind TensorFlow, Scikit-learn, and Pandas.
