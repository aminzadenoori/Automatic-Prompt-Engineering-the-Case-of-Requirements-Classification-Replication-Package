# Automatic Prompt Engineering: the Case of Requirements Classification (Replication Package)

This repository contains the implementation of the Automatic Prompt Engineering (APE) algorithm as described in the paper:

**Automatic Prompt Engineering: the Case of Requirements Classification**

*Authors:*
- **Mohammad Amin Zadenoori**, Consiglio Nazionale delle Ricerche ISTI “A. Faedo” (CNR-ISTI), Italy
  - Email: mohammadamin.zadenoori@isti.cnr.it
- **Alessio Ferrari**, Consiglio Nazionale delle Ricerche ISTI “A. Faedo” (CNR-ISTI), Italy
  - Email: alessio.ferrari@isti.cnr.it
- **Liping Zhao**, University of Manchester, UK
  - Email: liping.zhao@manchester.ac.uk
- **Waad Alhoshan**, Al-Imam Muhammad ibn Saud Islamic University, Saudi Arabia
  - Email: wmaboud@imamu.edu.sa

---

## Overview

This repository provides the code for implementing an iterative approach to automatically engineer prompts for classifying software requirements using Large Language Models (LLMs). The algorithm leverages misclassified samples from a validation set to refine prompts and improve classification performance over multiple iterations. The prompts used in this study are located in the file `Prompt.txt`.

The code evaluates multiple prompting methods:

- **Zero-Shot Prompting**
- **Chain-of-Thought (CoT) Prompting**
- **Few-Shot Prompting**
- **CoT with Examples**
- **Automatic Prompt Engineering (APE)**

---

## Requirements

- **Python 3.7+**
- **Libraries**:
  - `transformers`
  - `pandas`
  - `scikit-learn`
  - `numpy`
  - `torch`
  - `huggingface_hub`
- **Dataset**:
  - A CSV file named [Promise-Reclass.csv](https://github.com/explainable-re/RE-2019-Materials/blob/master/Manually%20tagged%20datasets/promise-reclass.csv) containing the software requirements data.
- **Hugging Face Access Token**:
  - A Hugging Face access token is required to use the LLM from the Hugging Face Hub.



---

## Setup and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/automatic-prompt-engineering.git
cd automatic-prompt-engineering
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

Ensure that your `requirements.txt` includes all the necessary libraries:

```txt
transformers
pandas
scikit-learn
numpy
torch
huggingface_hub
```

### 3. Prepare the Dataset

Place the `promise-reclass.csv` file in the root directory of the repository. This file should contain the software requirements data with the following columns:

- `RequirementText`: The text of the requirement.
- `IsFunctional`: Binary label indicating if the requirement is functional (`1`) or not (`0`).
- `IsQuality`: Binary label indicating if the requirement is quality-related (`1`) or not (`0`).

### 4. Obtain a Hugging Face Access Token

To use the LLM from Hugging Face, you need an access token.

- **Sign up** or **log in** to your Hugging Face account at [huggingface.co](https://huggingface.co/).
- Navigate to your [Access Tokens](https://huggingface.co/settings/tokens) page.
- Click on **"New token"** to create a new access token.
- **Copy** the token string.

### 5. Update the Code with Your Token

In the script `iterative_ape_classification.py`, replace the placeholder token with your Hugging Face token:

```python
# Log in to Hugging Face Hub
login(token="your_huggingface_token_here")
```

### 6. Run the Script

You can run the script directly:

```bash
python iterative_ape_classification.py
```

The script will perform the classification process, iteratively refining the prompts, and will output:

- The best classification metrics for each run, saved as CSV files.
- The best binary classification results for functionality and quality, saved as CSV files.

### 7. Understanding the Output

The script will produce CSV files containing the classification results and metrics:

- `best_classification_metrics_run1_best_f1.csv`: Contains the best classification metrics for the run.

---

## Algorithm Details

### Iterative Prompt Optimization

The algorithm performs an iterative process to optimize prompts:

1. **Initial Prompting**: Starts with initial examples selected from the dataset.
2. **Classification**: The LLM classifies the test set using different prompting methods.
3. **Misclassification Analysis**: Identifies misclassified samples from the validation set.
4. **Prompt Optimization**: Misclassified samples are used to update and optimize the prompt.
5. **Iteration**: The process repeats, integrating misclassified samples into the prompt.
6. **Stopping Condition**: Iterations continue until either:
   - The maximum number of iterations (`20`) is reached, or
   - The difference in F1-score is less than `0.02` compared to the best baseline.

### Prompting Methods

The script evaluates multiple prompting methods:

- **Zero-Shot Prompting**: The model is given the task without any examples.
- **Chain-of-Thought (CoT) Prompting**: The model is guided through a reasoning process.
- **Few-Shot Prompting**: The model is provided with examples of inputs and their corresponding outputs.
- **CoT with Examples**: Combines Chain-of-Thought with examples.
- **Automated Prompt Evolution (APE)**: The prompt is iteratively optimized using misclassified samples from the validation set.

---

## Code Structure

- **`iterative_ape_classification.py`**: The main script implementing the algorithm.
- **`promise-reclass.csv`**: The dataset file (not included; you need to provide it).
- **`requirements.txt`**: Contains the list of required Python libraries.

---

## Reproducing Results from the Paper

This code is associated with the paper:

**Automatic Prompt Engineering: the Case of Requirements Classification**

To run the code in this repository, follow these steps:

1. **Prerequisites**
   - Ensure you have Python installed on your system.
   - Install the required libraries with:
     ```bash
     pip install pandas sklearn transformers torch huggingface_hub
     ```

2. **Hugging Face Hub Authentication**
   - Obtain your Hugging Face Hub token and replace `"your_huggingface_token_here"` in the code.
   - This is required to load the model from Hugging Face.

3. **Run the Script**

   - **Step-by-Step Breakdown:**
     1. **Data Preparation**:
         - The script reads in `promise-reclass.csv` to create binary labels for functionality (`Functional` and `Non-Functional`) and quality (`Quality` and `Non-Quality`).
         - Initial samples are selected for both functionality and quality categories.
         - The remaining data is split into test and validation sets.

     2. **Model and Pipeline Setup**:
         - The Hugging Face `transformers` pipeline loads the `Meta-Llama-3-8B-Instruct` model, which is then used for generating text prompts.
         - Tokenization settings are configured for batching.

     3. **Prompt Generation and Classification**:
         - The code defines multiple prompt types (zero-shot, few-shot, chain-of-thought) to classify text based on requirements.
         - For functionality and quality classifications, optimized prompts are generated to improve classification accuracy.

     4. **Iterative Optimization and Evaluation**:
         - An iterative process optimizes prompts by incorporating examples from misclassified data, evaluating each iteration to find the best prompt.
         - Metrics such as F1-score, precision, recall, and AUC are calculated for each classification task.
         - The script stops further iterations if improvements in F1-score fall below a threshold.

     5. **Saving Results**:
         - The script saves the best-performing models and metrics as CSV files for each classification task.

4. **Running the Main Function**
   - Run the main function to initiate classification and save the results:
     ```python
     run_classification(df_uploaded, pipeline, num_runs=1)
     ```

5. **Output**
   - Results are saved in CSV format within the working directory, showing metrics and misclassified samples for further analysis.

This process automates the classification of functionality and quality requirements while iteratively optimizing prompts to enhance performance.

If you use this code in your research or wish to refer to the methodology, please cite the paper accordingly.

---
### Output Data (`output.csv`)

This file, `Output.csv`, contains the results of the classification task. It provides key classification metrics (Precision, Recall, F1, and F2 scores) for different  task categories. 

#### Columns:
Each column represents one of the primary metrics in classification performance:
- **P**: Precision - the proportion of true positives among predicted positives.
- **R**: Recall - the proportion of true positives among actual positives.
- **F1**: F1 Score - the harmonic mean of Precision and Recall, balancing the two.
- **F2**: F2 Score - a variant of the F-measure, which places more emphasis on Recall.

The columns are organized by category:
- **F**: Functional classification metrics
- **Q**: Quality classification metrics
- **onlyF**: Only functional classification 
- **onlyQ**: Only quality classification

#### Rows:
The rows represent the different classification methods used:
- **Zero-shot**: No task-specific training data was used.
- **Few-shot**: Limited examples were provided for training.
- **CoT**: Chain of Thought approach for task-solving steps.
- **CoT U Few-shot**: Combination of Chain of Thought and Few-shot methods.
- **APE-fixed (10, 20, 30)**:Fixed Automatic Prompt Engineering technique with traning set adjustment.
- **APE (10, 20, 30)**: Automatic Prompt Engineering versions with traning set adjustment.

This structure allows quick comparisons across classification strategies and categories, enabling insights into how each approach performs across both Functional and Quality-based tasks.


## Contact

For any questions or issues, please contact:

- **Mohammad Amin Zadenoori**: [mohammadamin.zadenoori@isti.cnr.it](mailto:mohammadamin.zadenoori@isti.cnr.it)
- **Alessio Ferrari**: [alessio.ferrari@isti.cnr.it](mailto:alessio.ferrari@isti.cnr.it)
- **Liping Zhao**: [liping.zhao@manchester.ac.uk](mailto:liping.zhao@manchester.ac.uk)
- **Waad Alhoshan**: [wmaboud@imamu.edu.sa](mailto:wmaboud@imamu.edu.sa)



