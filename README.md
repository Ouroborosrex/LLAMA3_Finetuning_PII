# LLM Experimentation: PII Instruction Tuning Dataset

## Description
This project focuses on the creation of a specialized dataset and training of a model for the detection of Personally Identifiable Information (PII). It utilizes two BERT-based datasets: the Kaggle competition dataset from The Learning Agency Lab - PII Data Detection, and the PII masking 300k dataset. The aim is to develop a model capable of identifying and correctly formatting PII data, enhancing privacy and data security measures.

## Installation
1. **Prerequisites**: Ensure Python 3.x is installed on your system.
2. **Dependencies**: Install necessary Python libraries using pip:
   ```bash
   pip install numpy pandas torch transformers
   ```
3. **Dataset**: Download the datasets from their respective sources and place them in the `data/` directory.

## Usage
1. **Prepare the Dataset**: 
   - Open and run the `dataset_instruction_gen.ipynb` notebook to generate the training dataset.
2. **Model Training**:
   - Execute the `fine_tune_llama3.ipynb` notebook to train the model using the prepared dataset.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Thanks to The Learning Agency Lab and contributors of the PII masking 300k dataset for providing the data used in this project.

---

You can customize each section according to the specifics of your project, such as installation steps, library versions, and how you organize your data and notebooks. Does this structure meet your needs? Would you like to add or modify any sections?
