import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import os

# --- Model Definition ---
class VulnerabilityTokenClassifier(nn.Module):
    def __init__(self, model, num_labels=2, dropout=0.1, class_weights=None):
        super().__init__()
        self.num_labels = num_labels
        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.class_weights = class_weights
        self.classifier = nn.Linear(model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss': loss, 'logits': logits}

# --- Aggregation Function ---


# --- Main Analyzer Class ---
class VulnerabilityAnalyzer:
    def __init__(self, model_path='model/model.pth', model_name="neulab/codebert-cpp", max_length=512):

        # Build absolute path to model file, relative to this script's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_model_path = os.path.join(base_dir, model_path)

         # Force CPU to avoid CUDA compatibility issues
        self.device = torch.device("cpu")
        print("Forcing CPU usage to ensure compatibility.")

        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load base model
        base_model = AutoModel.from_pretrained(model_name).to(self.device)

        # Load the full classifier model
        # The class weights are not needed for inference but were part of the model's training setup.
        class_weights = torch.tensor([0.3, 0.7], dtype=torch.float).to(self.device)
        self.model = VulnerabilityTokenClassifier(base_model, class_weights=class_weights).to(self.device)
        self.model.load_state_dict(torch.load(absolute_model_path, map_location=self.device))
        self.model.eval()
        print("VulnerabilityAnalyzer initialized and model loaded.")

    @staticmethod
    def _create_line_boundaries(text, offset_mapping):
        lines = text.split('\n')
        char_to_line = {}
        char_pos = 0
        for line_idx, line in enumerate(lines):
            for _ in range(len(line)):
                char_to_line[char_pos] = line_idx + 1
                char_pos += 1
            if line_idx < len(lines) - 1:
                char_to_line[char_pos] = line_idx + 1
                char_pos += 1

        line_boundaries = []
        offset_mapping = offset_mapping.squeeze(0)
        for start_char, end_char in offset_mapping:
            if start_char == 0 and end_char == 0:
                line_boundaries.append(-1)
            else:
                token_lines = set()
                for char_pos in range(start_char, end_char):
                    if char_pos in char_to_line:
                        token_lines.add(char_to_line[char_pos])
                if token_lines:
                    primary_line = min(token_lines)
                    line_boundaries.append(primary_line)
                else:
                    line_boundaries.append(-1)
        return line_boundaries
    
    @staticmethod
    def _aggregate_token_predictions_to_lines(token_preds, line_boundaries):
        line_to_token_preds = defaultdict(list)
        for pred, line in zip(token_preds, line_boundaries):
            if line > 0:  # ignore special tokens
                line_to_token_preds[line].append(pred)
        # Aggregate: line is vulnerable if any token is predicted vulnerable
        line_preds = {line: int(any(tokens)) for line, tokens in line_to_token_preds.items()}
        return line_preds
    

    def predict(self, code: str, threshold: float = 0.6):
        # Tokenize the input code
        tokenized = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        # Create line boundaries for tokens
        line_boundaries = self._create_line_boundaries(code, tokenized['offset_mapping'])

        # Move tensors to the correct device
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits'].cpu()
            probs = F.softmax(logits, dim=-1).numpy()
            # Use the threshold for the "vulnerable" class (class 1)
            token_preds = (probs[..., 1] > threshold).astype(int).squeeze(0)

        # Filter out special tokens
        valid_idx = [i for i, line in enumerate(line_boundaries) if line > 0]
        filtered_token_preds = token_preds[valid_idx]
        filtered_line_boundaries = [line_boundaries[i] for i in valid_idx]

        # Aggregate token predictions to line-level
        line_predictions = self._aggregate_token_predictions_to_lines(
            filtered_token_preds,
            filtered_line_boundaries
        )

        # Format the output
        total_lines = len(code.split('\n'))
        result = []
        for i in range(1, total_lines + 1):
            result.append({
                "line": i,
                "vulnerable": line_predictions.get(i, 0)
            })

        return result

# Example usage:
if __name__ == '__main__':
    analyzer = VulnerabilityAnalyzer()

    sample_code = """
    int main() {
        char name[20];
        printf("Enter name: ");
        gets(name); // Vulnerable line
        printf("Hello, %s\\n", name);
        return 0;
    }
    """

    predictions = analyzer.predict(sample_code)
    import json
    print(json.dumps(predictions, indent=2))