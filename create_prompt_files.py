import random

def process_dataset(input_files, dataset_name, label_col=None, sample_size=30):
    """
    Processes a dataset to generate both prompt and ground truth files.
    
    Args:
        input_files: List of input file paths
        dataset_name: Name for output files (e.g., "MR", "SemEval")
        label_col: For tab-separated files, column index for label
        sample_size: Number of samples to include
    """
    # Read and prepare data
    data = []
    
    for file_idx, input_file in enumerate(input_files):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if label_col is not None:
                    parts = line.split('\t')
                    if len(parts) >= label_col + 1:
                        label = parts[label_col].lower()
                        text = parts[-1]  # Text is last column
                        data.append((text, label))
                else:
                    label = 'positive' if file_idx == 0 else 'negative'
                    data.append((line, label))
    
    # Randomly sample data
    sampled_data = random.sample(data, min(sample_size, len(data)))
    
    # Generate prompt file
    with open(f"{dataset_name}-prompt.txt", 'w', encoding='utf-8') as f:
        f.write("Categorize the following texts as positive or negative. Give output in csv format, labels in lower case.\n")
        for text, _ in sampled_data:
            text_escaped = text.replace('"', '""').strip()
            f.write(f'"{text_escaped}"\n')
    
    # Generate ground truth file
    with open(f"{dataset_name}-ground_truth.txt", 'w', encoding='utf-8') as f:
        f.write("text,label\n")
        for text, label in sampled_data:
            text_escaped = text.replace('"', '""')
            f.write(f'"{text_escaped}",{label}\n')
    
# Process SemEval dataset
process_dataset(
    input_files=['datasets/Semeval2017A/gold/SemEval2017-task4-test.subtask-A.english.txt'],
    dataset_name='SemEval2017A',
    label_col=1,
    sample_size=30
)

# Process MR dataset
process_dataset(
    input_files=['datasets/MR/rt-polarity.pos', 'datasets/MR/rt-polarity.neg'],
    dataset_name='MR',
    sample_size=30
)