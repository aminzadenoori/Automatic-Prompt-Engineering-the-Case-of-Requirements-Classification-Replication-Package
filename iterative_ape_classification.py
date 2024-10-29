import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, fbeta_score, roc_auc_score
from sklearn.model_selection import train_test_split
import logging
import transformers
import torch
from huggingface_hub import login
import re
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log in to Hugging Face Hub
login(token="")

# Load the model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    return_full_text=False,
    do_sample=False  # Disable sampling for deterministic outputs
)

global functional_optmised_prompt 
global qaulity_optmised_prompt 
functional_optmised_prompt = ""
qaulity_optmised_prompt =""

# Set the pad_token_id to eos_token_id to enable batching
pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
pipeline.tokenizer.padding_side = "left"

# Load the data from the CSV file once
file_path = './promise-reclass.csv'
df_uploaded = pd.read_csv(file_path)

# Define binary labels for functionality and quality classification
df_uploaded['Functional'] = df_uploaded['IsFunctional'].apply(lambda x: 'Functional' if x == 1 else 'Non-Functional')
df_uploaded['Quality'] = df_uploaded['IsQuality'].apply(lambda x: 'Quality' if x == 1 else 'Non-Quality')

# Define terminators
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Function to prepare data
def prepare_data(df):
    # Initial sample selection: 4 examples of each class for functionality
    functional_examples = df.groupby('Functional').apply(lambda x: x.sample(min(4, len(x)), random_state=42)).reset_index(drop=True)
    df_remaining_functional = df.drop(functional_examples.index)

    # Initial sample selection: 4 examples of each class for quality
    quality_examples = df.groupby('Quality').apply(lambda x: x.sample(min(4, len(x)), random_state=42)).reset_index(drop=True)
    df_remaining_quality = df.drop(quality_examples.index)

    # Split remaining data into 70% test and 30% validation for functionality
    df_functional_train_test = df_remaining_functional[['RequirementText', 'Functional']].copy()
    df_functional_train_test.columns = ['text', 'label']
    df_functional_val, df_functional_test = train_test_split(df_functional_train_test, test_size=0.7, random_state=42, stratify=df_functional_train_test['label'])

    # Split remaining data into 70% test and 30% validation for quality
    df_quality_train_test = df_remaining_quality[['RequirementText', 'Quality']].copy()
    df_quality_train_test.columns = ['text', 'label']
    df_quality_val, df_quality_test = train_test_split(df_quality_train_test, test_size=0.7, random_state=42, stratify=df_quality_train_test['label'])

    return functional_examples, quality_examples, df_functional_val.reset_index(drop=True), df_functional_test.reset_index(drop=True), df_quality_val.reset_index(drop=True), df_quality_test.reset_index(drop=True)

# Function to generate prompts
def generate_prompts(df, class_examples, class_explanations, is_quality=False, optimized_prompt=None):
    # Prepare examples and explanations
    example_str = "\n".join([f'"{text}" --> {label}' for text, label in class_examples.items()])
    explanation_str = "\n".join([f'{label}: {explanation}' for label, explanation in class_explanations.items()])

    # Generate prompts using the chat template
    def generate_messages(text, examples=None, explanations=None, optimized_prompt=None):
        if not is_quality:
            system_content = """
As an expert system for classifying software requirements, your task is to carefully analyze each given requirement and categorize it into one of the following two categories:

"1": Functional
"2": Non-Functional

Output only the number (1 or 2) that corresponds to the appropriate category. Do not provide any additional explanations.

Please provide the categorized number for the given software requirement, with no additional text.
"""
        else:
            system_content = """
As an expert system for classifying software requirements, your task is to carefully analyze each given requirement and categorize it into one of the following two categories:

"1": Quality
"2": Non-Quality

Output only the number (1 or 2) that corresponds to the appropriate category. Do not provide any additional explanations.

Please provide the categorized number for the given software requirement, with no additional text.
"""

        messages = [{"role": "system", "content": system_content.strip()}]

        if optimized_prompt:
            content = f"""
{optimized_prompt}

Requirement: {text}
"""
        elif examples and explanations:
            content = f"""
Let's analyze the classification of requirements step by step.

Step 1: Review the examples of different types of requirements and their classifications:

{examples}

Step 2: Read the explanations for these classifications:

{explanations}

Step 3: Understand how to classify requirements using the examples and explanations as guidance.

Step 4: Apply this understanding to the following requirement:

Requirement: {text}

Step 5: Determine the classification of the requirement and provide the final label of the class without any explanations.
"""
        elif examples:
            content = f"""
Below are examples of different types of requirements and their classifications:

{examples}

Requirement: {text}
"""
        elif explanations:
            content = f"""
Let's analyze the classification of the requirement step by step.

Step 1: Read the explanations of different types of requirements:

{explanations}

Step 2: Understand how to classify requirements using these explanations as guidance.

Step 3: Apply this understanding to the following requirement:

Requirement: {text}

Step 4: Determine the classification of the requirement and provide the final label of the class without any explanations.
"""
        else:
            content = f"""
Requirement: {text}
"""
        messages.append({"role": "user", "content": content.strip()})
        return messages

    def generate_all_prompts(texts, examples=None, explanations=None, optimized_prompt=None):
        prompts = []
        for text in texts:
            messages = generate_messages(text, examples, explanations, optimized_prompt)
            prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        return prompts

    zero_shot_prompts = generate_all_prompts(df['text'].tolist())
    COT_prompts = generate_all_prompts(df['text'].tolist(), explanations=explanation_str)
    few_shot_prompts = generate_all_prompts(df['text'].tolist(), examples=example_str)
    COT_with_examples_prompts = generate_all_prompts(df['text'].tolist(), examples=example_str, explanations=explanation_str)
    ape_prompts = generate_all_prompts(df['text'].tolist(), optimized_prompt=optimized_prompt)

    return zero_shot_prompts, COT_prompts, few_shot_prompts, COT_with_examples_prompts, ape_prompts

# Function to process batches
def process_batch(prompts):
    batch_size = 4
    predictions = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        outputs = pipeline(
            batch,
            max_new_tokens=50,
            eos_token_id=terminators,
            batch_size=batch_size
        )
        outputs = remove_assistant_prefix(outputs)
        predictions.extend([output[0]['generated_text'] for output in outputs])
    return predictions

# Mapping function
def mapping(outputs, labels):
    mapped = []
    for output in outputs:
        output_lower = output.strip().lower()
        found = False
        sorted_labels = sorted(labels.items(), key=lambda x: -len(x[0]))
        for label_key, label_value in sorted_labels:
            pattern = re.compile(r'\b' + re.escape(label_key.lower()) + r'\b')
            if pattern.search(output_lower):
                mapped.append(label_value)
                found = True
                break
        if not found:
            mapped.append('Unknown')
    return mapped

# Function to clean up the model's generated text
def remove_assistant_prefix(data):
    cleaned_data = []
    for item in data:
        text = item[0]['generated_text']
        if text.startswith('assistant\n\n'):
            text = text[11:]
        cleaned_data.append([{'generated_text': text}])
    return cleaned_data

import re

def extract_steps(text, examples):
    """
    Extracts steps from the given text starting from the beginning up to 'Step 5', and replaces
    the content between 'Step 1' and 'Step 2' with a specific phrase containing examples.

    Parameters:
    text (str): The input text containing steps and other content.
    examples (str): The examples to be inserted between Step 1 and Step 2.

    Returns:
    str: A string containing the modified steps extracted from the input text.
    """
    # Match content from the beginning of the text up to 'Step 5:' (inclusive)
    pattern = r'(.*?)(Step 1:)(.*?)(Step 2:.*?Step 5:.*?(?:\n|$))'

    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # match.group(1): Content before 'Step 1:'
        # match.group(2): 'Step 1:'
        # match.group(3): Content between 'Step 1:' and 'Step 2:'
        # match.group(4): Content from 'Step 2:' up to 'Step 5:'
        
        # Replace the content between 'Step 1:' and 'Step 2:' with the provided examples
        replacement_text = f"{match.group(2)}\nReview the examples of different types of requirements and their classifications:\n\n{examples}\n\n"
        
        # Combine all parts to form the modified text
        modified_text = match.group(1) + replacement_text + match.group(4)
        # Split the text into lines
        lines = modified_text.split('\n')
        
        # Remove the first line
        remaining_lines = lines[1:]
        
        # Join back into a single string
        modified_text = '\n'.join(remaining_lines)
        
        return modified_text
    else:
        return ''
    
# Function to generate an optimized classification prompt
def generate_optimized_classification_prompt(examples, labels, class_explanations, is_quality=False):
    
    formatted_examples = "\n".join([f'"{text}" --> {label}' for text, label in zip(examples, labels)])
    explanations = "\n".join([f"- {label}: {explanation}" for label, explanation in class_explanations.items()])
    category = "Quality" if is_quality else "Functional"
    opposite_category = "Non-Quality" if is_quality else "Non-Functional"
    
    sample_prompt = f"""
You are an expert system that needs to classify software requirements into two categories: {category} and {opposite_category}.

Let's analyze the classification of requirements step by step.

Step 1: Review the examples of different types of requirements and their classifications:

{formatted_examples}

Step 2: Read the explanations for these classifications:

{explanations}

Step 3: Understand how to classify requirements using the examples and explanations as guidance.

Step 4: Apply this understanding to the following requirement.

Step 5: Determine the classification of the requirement and provide the final label of the class without any explanations.
Based on these examples and explanations classify unseen software requirements into {category} or {opposite_category}. Just give the final label without any explanations. The output categories should be exactly the same as the categories mentioned here.
"""
    global functional_optmised_prompt 
    global qaulity_optmised_prompt 

    if is_quality:
        if qaulity_optmised_prompt == "":
            final_prompt = sample_prompt
        else:
            final_prompt = extract_steps(qaulity_optmised_prompt, formatted_examples)
    

    else:
        if functional_optmised_prompt == "":
            final_prompt = sample_prompt
        else:
            final_prompt = extract_steps(functional_optmised_prompt, formatted_examples)
    

    # Optimization prompt with instructions to create a generalized prompt without examples
    
    optimization_prompt = f'''
    You are required to enhance and clarify the explanations of the categories in the prompt by integrating illustrative examples and information implicitly referenced in the initial context. The optimized prompt must follow these strict guidelines:
    
    Maintain the Original Steps: The steps in the optimized prompt must remain exactly the same as in the sample prompt; no changes should be made to the steps' structure or order. Do not add any new steps or content beyond the existing steps.
    Expand Explanations: Enrich and expand the explanations of each category within the steps, incorporating any new examples provided at the end of the sample prompt. Use these examples to enhance understanding and provide clarity, but ensure all content remains within the existing steps and does not extend beyond them.
    Incorporate Class Explanations: Specifically, integrate the detailed "Class Explanations" of categories from the first prompt into the optimized prompt. For each category, introduce implicit clarifications based on relevant data extracted from the context, keeping all additions within the boundaries of the original steps.
    Use New Examples: If there are new examples at the end of the prompt, use them to further expand and illustrate the explanations within the existing steps. Do not include any content beyond step 5.
    End Strictly After Step 5: The optimized prompt must strictly end after step 5. Do not add any additional steps, conclusions, or content beyond this point.
    Focus Only on Explanations: Remember, only the explanations within the steps should be expanded; the steps themselves should remain unchanged in structure and order. Any new examples should be used to enhance the explanations within the steps, not to add new content or extend the prompt beyond its original end. 
    Given the above strict instructions, extend the prompt below using the outlined techniques. Ensure that the new optimized prompt ends strictly after step 5, with no additional content beyond that point:
    """
    {final_prompt}
    """
    '''
    
    print(optimization_prompt)


    systemprompt = "Imagine yourself as an expert in the realm of prompting techniques for LLMs. Your expertise is not just broad, encompassing the entire spectrum of current knowledge on the subject, but also deep, delving into the nuances and intricacies that many overlook. Your job is to reformulate prompts with surgical precision, optimizing them for the most accurate response possible. The reformulated prompt should enable the LLM to always give the correct answer to the question."

    # Generate the optimized prompt
    optimized_prompt, _ = generate_response(optimization_prompt,systemprompt)
    
    if  is_quality:
        qaulity_optmised_prompt = optimized_prompt
    else:
        functional_optmised_prompt = optimized_prompt
    optimized_prompt=extract_steps(optimized_prompt,formatted_examples)
    print(optimized_prompt)
    return optimized_prompt.strip()

# Function to generate response using the model
def generate_response(prompt, systemprompt):
    full_prompt = f"{systemprompt}\n\n{prompt}"
    try:
        completion = pipeline(full_prompt, max_new_tokens=1000, num_return_sequences=1)
        return completion[0]['generated_text'].strip(), full_prompt
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        return "", full_prompt

# Function to compute classification metrics
def compute_classification_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=[0,1], output_dict=True, zero_division=0)
    metrics = {}
    metrics['precision'] = report['1']['precision']
    metrics['recall'] = report['1']['recall']
    metrics['f1-score'] = report['1']['f1-score']
    metrics['support'] = report['1']['support']
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f2-score'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    # Compute AUC if possible
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = np.nan
    metrics['AUC'] = auc
    return metrics

# Main function to run the process
def run_classification(df_uploaded, pipeline, num_runs=1):
    # Initialize list to collect metrics for each run
    all_metrics = []
    global optmised_prompt 
    optmised_prompt = ""
    
    # Define labels with potential variations
    labels_functional = {
        'Non-Functional': 'Non-Functional',
        'Functional': 'Functional',
        '2': 'Non-Functional',
        '1': 'Functional'
    }

    labels_quality = {
        'Non-Quality': 'Non-Quality',
        'Quality': 'Quality',
        '2': 'Non-Quality',
        '1': 'Quality'
    }

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        
        # Prepare data
        functional_examples, quality_examples, df_functional_val, df_functional_test, df_quality_val, df_quality_test = prepare_data(df_uploaded)

        # Prepare examples and explanations for functionality classification
        class_explanations_functional = {
            'Functional': """Functional requirements specify what the system should do. They describe the functionality or services that the system is expected to provide. These requirements define the specific behaviors or functions of the system, such as calculations, technical details, data manipulation, processing, and other specific functionality that shows how the system is to perform certain tasks.""",
            'Non-Functional': """Non-functional requirements specify how the system should behave and impose constraints on the system's functionality. They describe the system's qualities or attributes, such as performance, security, usability, reliability, etc. These requirements define criteria that can be used to judge the operation of a system, rather than specific behaviors."""
        }

        # Prepare examples and explanations for quality classification
        class_explanations_quality = {
            'Quality': """Quality requirements specify attributes like performance, security, usability, reliability, etc. These requirements focus on how well the system performs its functions and impose constraints on the system's operation.""",
            'Non-Quality': """Non-quality requirements are requirements that do not specify quality attributes. They might include functional requirements or other specifications that do not focus on the quality aspects of the system."""
        }

        # Initialize variables to keep track of the best results
        best_f1_scores = {
            'Functional': 0,
            'Quality': 0,
            'Functional_NonQuality_vs_All': 0,
            'Quality_NonFunctional_vs_All': 0
        }
        best_metrics = {
            'Functional': None,
            'Quality': None,
            'Functional_NonQuality_vs_All': None,
            'Quality_NonFunctional_vs_All': None
        }
        best_iteration = {
            'Functional': 0,
            'Quality': 0,
            'Functional_NonQuality_vs_All': 0,
            'Quality_NonFunctional_vs_All': 0
        }
        # To store best DataFrames
        best_df_functional_test = None
        best_df_quality_test = None

        # Iterative process for few-shot-with and APE
        max_iterations = 20
        f1_score_threshold = 0.02
        previous_f1_scores = {
            'Functional': 0,
            'Quality': 0,
            'Functional_NonQuality_vs_All': 0,
            'Quality_NonFunctional_vs_All': 0
        }

        for iteration in range(max_iterations):
            print(f"Iteration {iteration+1}/{max_iterations}")

            # For the first iteration, use initial examples
            if iteration == 0:
                # Functionality classification examples
                class_examples_functional = functional_examples.set_index('RequirementText')['Functional'].to_dict()
                train_texts_f = functional_examples['RequirementText'].tolist()
                train_labels_f = functional_examples['Functional'].tolist()
            
                # Quality classification examples
                class_examples_quality = quality_examples.set_index('RequirementText')['Quality'].to_dict()
                train_texts_q = quality_examples['RequirementText'].tolist()
                train_labels_q = quality_examples['Quality'].tolist()
            else:
                # For functionality classification
                # Generate the optimized classification prompt using the previous examples
                optimized_classification_prompt_f = generate_optimized_classification_prompt(train_texts_f, train_labels_f, class_explanations_functional, is_quality=False)
            
                # Generate prompts for the validation set
                _, _, _, _, ape_prompts_val_f = generate_prompts(
                    df_functional_val, class_examples_functional, class_explanations_functional, optimized_prompt=optimized_classification_prompt_f
                )
            
                # Process batches and get predictions for the validation set
                outputs_val_f = process_batch(ape_prompts_val_f)
                preds_val_f = mapping(outputs_val_f, labels_functional)
                df_functional_val['Predicted_Label'] = preds_val_f
            
                # Identify misclassified samples
                misclassified_f = df_functional_val[df_functional_val['label'] != df_functional_val['Predicted_Label']]
            
                # Select misclassified samples to use as new examples
                misclassified_examples_f = misclassified_f.groupby('label').apply(lambda x: x.sample(min(4, len(x)), random_state=iteration)).reset_index(drop=True)
                class_examples_functional = misclassified_examples_f.set_index('text')['label'].to_dict()
                train_texts_f = list(class_examples_functional.keys())
                train_labels_f = list(class_examples_functional.values())

                # For quality classification
                optimized_classification_prompt_q = generate_optimized_classification_prompt(train_texts_q, train_labels_q, class_explanations_quality, is_quality=True)

                # Generate prompts for the validation set
                _, _, _, _, ape_prompts_val_q = generate_prompts(
                    df_quality_val, class_examples_quality, class_explanations_quality, is_quality=True, optimized_prompt=optimized_classification_prompt_q
                )

                # Process batches and get predictions for the validation set
                outputs_val_q = process_batch(ape_prompts_val_q)
                preds_val_q = mapping(outputs_val_q, labels_quality)
                df_quality_val['Predicted_Label'] = preds_val_q

                # Identify misclassified samples
                misclassified_q = df_quality_val[df_quality_val['label'] != df_quality_val['Predicted_Label']]

                # Select misclassified samples to use as new examples
                misclassified_examples_q = misclassified_q.groupby('label').apply(lambda x: x.sample(min(4, len(x)), random_state=iteration)).reset_index(drop=True)
                class_examples_quality = misclassified_examples_q.set_index('text')['label'].to_dict()
                train_texts_q = list(class_examples_quality.keys())
                train_labels_q = list(class_examples_quality.values())

            # Generate the optimized classification prompt for functionality
            optimized_classification_prompt_f = generate_optimized_classification_prompt(train_texts_f, train_labels_f, class_explanations_functional, is_quality=False)

            # Generate prompts for functionality classification
            zero_shot_prompts_f, COT_prompts_f, few_shot_prompts_f, COT_with_examples_prompts_f, ape_prompts_f = generate_prompts(
                df_functional_test, class_examples_functional, class_explanations_functional, optimized_prompt=optimized_classification_prompt_f
            )

            # Process batches and get predictions for functionality
            predictions_f = {}
            for method, prompts in zip(['ZeroShot', 'COT', 'FewShot', 'COT_with_examples', 'APE'],
                                       [zero_shot_prompts_f, COT_prompts_f, few_shot_prompts_f, COT_with_examples_prompts_f, ape_prompts_f]):
                outputs = process_batch(prompts)
                preds = mapping(outputs, labels_functional)
                predictions_f[method] = preds
                df_functional_test[method] = preds

            # Generate the optimized classification prompt for quality
            optimized_classification_prompt_q = generate_optimized_classification_prompt(train_texts_q, train_labels_q, class_explanations_quality, is_quality=True)

            # Generate prompts for quality classification
            zero_shot_prompts_q, COT_prompts_q, few_shot_prompts_q, COT_with_examples_prompts_q, ape_prompts_q = generate_prompts(
                df_quality_test, class_examples_quality, class_explanations_quality, is_quality=True, optimized_prompt=optimized_classification_prompt_q
            )

            # Process batches and get predictions for quality
            predictions_q = {}
            for method, prompts in zip(['ZeroShot', 'COT', 'FewShot', 'COT_with_examples', 'APE'],
                                       [zero_shot_prompts_q, COT_prompts_q, few_shot_prompts_q, COT_with_examples_prompts_q, ape_prompts_q]):
                outputs = process_batch(prompts)
                preds = mapping(outputs, labels_quality)
                predictions_q[method] = preds
                df_quality_test[method] = preds

            # Combine the functionality and quality DataFrames on 'text'
            combined_df = pd.merge(
                df_quality_test[['text', 'label', 'ZeroShot', 'COT', 'FewShot', 'COT_with_examples', 'APE']],
                df_functional_test[['text', 'label', 'ZeroShot', 'COT', 'FewShot', 'COT_with_examples', 'APE']],
                on='text',
                suffixes=('_Quality', '_Functional')
            )

            # Create the true combined labels
            combined_df['True_Label'] = combined_df['label_Quality'] + '-' + combined_df['label_Functional']

            # Create binary labels for 'Functional Non-Quality' vs All and 'Quality Non-Functional' vs All
            combined_df['True_Label_Functional_NonQuality'] = ((combined_df['label_Functional'] == 'Functional') & (combined_df['label_Quality'] == 'Non-Quality')).astype(int)
            combined_df['True_Label_Quality_NonFunctional'] = ((combined_df['label_Functional'] == 'Non-Functional') & (combined_df['label_Quality'] == 'Quality')).astype(int)

            # Methods to process
            methods = ['ZeroShot', 'COT', 'FewShot', 'COT_with_examples', 'APE']

            # Initialize list to collect metrics
            metrics_list = []

            for method in methods:
                # Ensure that there are no missing predicted labels for the method
                combined_df_method = combined_df.dropna(subset=[f'{method}_Quality', f'{method}_Functional'])

                # Create the predicted combined labels for the method
                combined_df_method['Predicted_Label'] = combined_df_method[f'{method}_Quality'] + '-' + combined_df_method[f'{method}_Functional']

                # Binary predicted labels for new classifications
                combined_df_method[f'Predicted_Label_Functional_NonQuality'] = ((combined_df_method[f'{method}_Functional'] == 'Functional') & (combined_df_method[f'{method}_Quality'] == 'Non-Quality')).astype(int)
                combined_df_method[f'Predicted_Label_Quality_NonFunctional'] = ((combined_df_method[f'{method}_Functional'] == 'Non-Functional') & (combined_df_method[f'{method}_Quality'] == 'Quality')).astype(int)

                # Compute metrics for 'Functional' classification
                y_true_f = df_functional_test['label'].map({'Functional':1, 'Non-Functional':0})
                y_pred_f = df_functional_test[method].map({'Functional':1, 'Non-Functional':0})
                metrics_f = compute_classification_metrics(y_true_f, y_pred_f)
                metrics_f['Classification'] = 'Functional'
                metrics_f['Method'] = method
                metrics_list.append(metrics_f)

                # Compute metrics for 'Quality' classification
                y_true_q = df_quality_test['label'].map({'Quality':1, 'Non-Quality':0})
                y_pred_q = df_quality_test[method].map({'Quality':1, 'Non-Quality':0})
                metrics_q = compute_classification_metrics(y_true_q, y_pred_q)
                metrics_q['Classification'] = 'Quality'
                metrics_q['Method'] = method
                metrics_list.append(metrics_q)

                # Compute metrics for 'Functional_NonQuality_vs_All' classification
                y_true_fnq = combined_df_method['True_Label_Functional_NonQuality']
                y_pred_fnq = combined_df_method[f'Predicted_Label_Functional_NonQuality']
                metrics_fnq = compute_classification_metrics(y_true_fnq, y_pred_fnq)
                metrics_fnq['Classification'] = 'Functional_NonQuality_vs_All'
                metrics_fnq['Method'] = method
                metrics_list.append(metrics_fnq)

                # Compute metrics for 'Quality_NonFunctional_vs_All' classification
                y_true_qnf = combined_df_method['True_Label_Quality_NonFunctional']
                y_pred_qnf = combined_df_method[f'Predicted_Label_Quality_NonFunctional']
                metrics_qnf = compute_classification_metrics(y_true_qnf, y_pred_qnf)
                metrics_qnf['Classification'] = 'Quality_NonFunctional_vs_All'
                metrics_qnf['Method'] = method
                metrics_list.append(metrics_qnf)

            # Convert metrics list to DataFrame
            metrics_df = pd.DataFrame(metrics_list)

            # Check and update best f1-scores for each classification task for all methods
            for method in methods:
                for classification in ['Functional', 'Quality', 'Functional_NonQuality_vs_All', 'Quality_NonFunctional_vs_All']:
                    method_metrics = metrics_df[(metrics_df['Method'] == method) & (metrics_df['Classification'] == classification)]
                    current_f1_score = method_metrics['f1-score'].values[0]
                    if current_f1_score > best_f1_scores[classification]:
                        best_f1_scores[classification] = current_f1_score
                        best_metrics[classification] = method_metrics.copy()
                        best_iteration[classification] = iteration + 1  # iterations are 0-indexed
                        if classification in ['Functional', 'Functional_NonQuality_vs_All']:
                            best_df_functional_test = df_functional_test.copy()
                        if classification in ['Quality', 'Quality_NonFunctional_vs_All']:
                            best_df_quality_test = df_quality_test.copy()

            # Add stopping condition based on F1-score increase for APE method
            # Initialize a flag to check whether to stop iterating
            stop_iteration = False

            # For each classification task, check the increase in F1-score for the APE method
            for classification in ['Functional', 'Quality', 'Functional_NonQuality_vs_All', 'Quality_NonFunctional_vs_All']:
                ape_metrics = metrics_df[(metrics_df['Method'] == 'APE') & (metrics_df['Classification'] == classification)]
                current_f1_score = ape_metrics['f1-score'].values[0]
                previous_f1_score = previous_f1_scores[classification]
                increase_in_f1 = current_f1_score - previous_f1_score

                # Update previous_f1_scores
                previous_f1_scores[classification] = current_f1_score

                # Check if increase is less than threshold
                if increase_in_f1 < f1_score_threshold:
                    stop_iteration = True

            if stop_iteration:
                print("Stopping iterations as the increase in F1-score is less than the threshold.")
                break

            # Update validation set with predictions for next iteration
            # This step is already done above when we assigned 'Predicted_Label' in df_functional_val and df_quality_val

        # ---------------------------------------------
        # Save the best metrics for each classification task
        # ---------------------------------------------
        # Concatenate all best metrics into a single DataFrame
        best_metrics_df = pd.concat(best_metrics.values(), ignore_index=True)
        best_metrics_df.to_csv(f'best_classification_metrics_run{run+1}_best_f1.csv', index=False)

        # Save the best binary classification results for functionality and quality
        if best_df_functional_test is not None:
            best_df_functional_test.to_csv(f'best_binary_classification_results_functionality_run{run+1}.csv', index=False)
        if best_df_quality_test is not None:
            best_df_quality_test.to_csv(f'best_binary_classification_results_quality_run{run+1}.csv', index=False)

    print("All experiment results have been saved.")

# Run the classification process and save results
run_classification(df_uploaded, pipeline, num_runs=1)
