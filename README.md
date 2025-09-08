# Code Completion with StarCoder

A Python implementation of an AI-powered code completion system using the StarCoder base model with LoRA fine-tuning. This project provides a lightweight yet powerful code completion capability that can be trained on custom datasets.

##  Features

- **Fill-in-the-Middle (FIM) Capability**: Handles both prefix-suffix code completion and middle-context completion
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using Low-Rank Adaptation
- **Modular Architecture**: Clean separation between settings, model components, and training logic
- **Customizable Training**: Easily adjust hyperparameters through the settings file
- **Apple Silicon Support**: Optimized for running on Apple MPS devices / Note-Since currenlty huggingface does'nt support MLX backend out of the box, hence training runs on mlx backend is slow when compare to cuda device.

##  Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- Accelerate
- BitsAndBytes (for quantization)

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/deep-learner-ConfigurableAI/code-completion.git
cd code-completion

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers datasets accelerate bitsandbytes peft tqdm
```

##  Project Structure

```
code-completion/
├── LICENSE
├── README.md
├── src/
│   ├── main.py           # Entry point for the application
│   ├── settings.py       # Configuration settings
│   ├── model.py          # Core model implementation
│   └── runner.py         # Training and inference logic
```

##  Configuration

All model and training configurations are centralized in `src/settings.py`. Key parameters include:

- Model checkpoint (`MODEL`)
- Training dataset (`DATASET`)
- Sequence length (`SEQ_LENGTH`)
- Training parameters (batch size, learning rate, etc.)
- LoRA configuration (rank, alpha, target modules)
- FIM transformation settings

##  Usage

### Training

To train the model on your dataset:

1. Update the `settings.py` file with your desired configuration
2. Uncomment the `train_model()` line in `main.py`
3. Run the following command:

```bash
cd src
python main.py
```

### Inference

To use the model for code completion:

1. Ensure you have a trained model or use the provided checkpoint
2. Uncomment the `code_completion_demo()` line in `main.py`
3. Run:

```bash
cd src
python main.py
```

### Custom Inference

You can also use the model programmatically:

```python
from model import load_model_tokenizer, get_code_completion
from runner import load_model_for_inference

# Load model
model, tokenizer = load_model_for_inference()

# Example code completion
prefix = "def calculate_total(items):"
suffix = "    return total"
completed_code = get_code_completion(model, tokenizer, prefix, suffix)
print(completed_code)
```

##  How It Works

1. **Fill-in-the-Middle (FIM)**: The model is trained to predict missing code in the middle of two context pieces (prefix and suffix).

2. **LoRA Fine-tuning**: Instead of fine-tuning all parameters, we use LoRA to efficiently adapt the pre-trained StarCoder model.

3. **Dataset Processing**: The training process formats the dataset into fixed-length chunks with FIM transformations applied.

4. **Constant Length Dataset**: For efficient training, we process examples into a constant length format.

## 📊 Performance

The model's performance depends on:
- The quality and size of the training dataset
- The hyperparameters used (especially LoRA rank and learning rate)
- The number of training steps

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- [BigCode Project](https://www.bigcode-project.org/) for the StarCoder base model
- [Hugging Face](https://huggingface.co/) for their excellent Transformers library
- [PEFT](https://github.com/huggingface/peft) for the efficient fine-tuning implementation
