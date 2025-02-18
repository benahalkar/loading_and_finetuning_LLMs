# Loading and Fine-tuning Large Language Models (LLMs)

This repository contains a Jupyter Notebook demonstrating how to load and fine-tune Large Language Models (LLMs) using Python and relevant libraries.

## Overview

The notebook provides a step-by-step guide on:

*   **Loading Pre-trained LLMs:**  Instructions on how to load LLMs from various sources like Hugging Face Model Hub.
*   **Preparing Data:**  Techniques for preparing and formatting data for fine-tuning.
*   **Fine-tuning LLMs:**  Methods for fine-tuning LLMs on custom datasets using libraries like Transformers.
*   **Evaluating Performance:**  Guidance on evaluating the performance of fine-tuned models.

## Repository Structure

```
.
├── README.md # This file
├── main.ipynb # Jupyter Notebook with the code
```


You can read more about the llama-2 model [here](https://ai.meta.com/llama/).<br>
<img src="./images/llama_2b.png" width="500">
<br><br>

This project utilizes the [Hugging Face API](https://huggingface.co/) to load the model and also demonstrates how to save it.<br>
<img src="./images/hugging_face.png" width="500">
<br><br>

[Tensorboard](https://www.tensorflow.org/tensorboard) was utilized to visualize the training loss.
<img src="./images/tensorboard.png" width="500">
<br><br>


Steps involved in this project.<br>
* Load the *Llama-2-7b* LLM from hugging face.<br>
* Load the *guanaco-llama2-1k* dataset to finetune the LLM.<br>
* Define the model and fine-tuning parameters.<br>
* Train the model on the new parameters.<br>
* Utilize *tensorboard* to visualize the training loss.<br>
* Push the new model to a user's hugging face account.<br>
* Test out the new model by providing a prompt.<br> 


## Requirements

Before running the notebook, ensure you have the following libraries installed:

*   transformers
*   torch
*   datasets
*   accelerate
*   evaluate
*   scikit-learn
*   (Add any other libraries used)

You can install these using pip:

```
pip install transformers torch datasets accelerate evaluate scikit-learn
```


It's highly recommended to use a virtual environment to manage dependencies.

## Usage

1.  **Clone the repository:**

    ```
    git clone https://github.com/benahalkar/loading_and_finetuning_LLMs.git
    cd loading_and_finetuning_LLMs
    ```

2.  **Install the requirements:**

    ```
    pip install -r requirements.txt # If you have a requirements.txt file
    ```
    # OR
    ```
    pip install transformers torch datasets accelerate evaluate scikit-learn
    ```

3.  **Open and run the Jupyter Notebook:**

    ```
    jupyter notebook main.ipynb
    ```

    Follow the instructions within the notebook to load, prepare, fine-tune, and evaluate your LLM.

## Jupyter Notebook Details

The `main.ipynb` notebook is structured as follows:

1.  **Introduction:**  A brief introduction to LLMs and the purpose of fine-tuning.
2.  **Loading the Model:**  Code demonstrating how to load a pre-trained LLM from Hugging Face Hub (e.g., `bert-base-uncased`, `gpt2`, or any other relevant model).  This section explains how to use `AutoModelForSequenceClassification`, `AutoTokenizer`, and other relevant classes from the `transformers` library.
3.  **Data Preparation:**  Code for loading and pre-processing your dataset.  This section covers:
    *   Loading data from a CSV file (or other formats).
    *   Tokenization using the loaded tokenizer.
    *   Creating training and validation datasets.
4.  **Fine-tuning the Model:**  Implementation of the fine-tuning process.  This section details:
    *   Setting up training arguments (learning rate, batch size, number of epochs, etc.).
    *   Using the `Trainer` class from `transformers` to train the model.
    *   (Optional) Implementing custom training loops.
5.  **Evaluation:**  Code for evaluating the fine-tuned model on a validation dataset.  This includes:
    *   Calculating metrics like accuracy, precision, recall, and F1-score.
    *   Generating classification reports.
6.  **Saving the Model:** Instructions on saving the fine-tuned model for later use.
7.  **Inference:** Demonstrates how to load the fine-tuned model and use it for making predictions on new data.

## Datasets

The `data/` directory (if present) contains the datasets used for fine-tuning and evaluation. Ensure that the datasets are properly formatted and compatible with the notebook's data loading code.  Describe the format of the datasets here.

## Fine-tuning Details

*   **Model:** The specific LLM used in the notebook (e.g., `bert-base-uncased`, `gpt2`).  Explain why this model was chosen.
*   **Dataset:**  The dataset used for fine-tuning (e.g., a custom dataset of text messages for sentiment analysis).  Provide details about the dataset's size, format, and source.
*   **Training Parameters:**  The key training parameters used during fine-tuning, such as:
    *   Learning rate
    *   Batch size
    *   Number of epochs
    *   Optimizer
    *   Loss function
*   **Evaluation Metrics:**  The metrics used to evaluate the performance of the fine-tuned model (e.g., accuracy, precision, recall, F1-score).

## Results

This section should present the results of fine-tuning. Include key metrics and observations about the model's performance.  You can include tables or visualizations to illustrate the results.

## Contributing

Contributions are welcome!  If you find any issues or have suggestions for improvements, please submit a pull request.

## License

[MIT License](LICENSE)

