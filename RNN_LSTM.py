"""Text classification using a simple RNN and an LSTM in PyTorch.

Run:
    pip install torch
    python -X utf8 Text_Classification_RNN_LSTM.py

Both models learn the same small, fictional sentiment dataset.
Label 0 = Negative, label 1 = Positive.
This is a classroom demonstration, not a production sentiment classifier.
"""

import random
import re

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

SAMPLES = [
    ("I love this movie", 1),
    ("This film was excellent", 1),
    ("The acting was wonderful", 1),
    ("What a fantastic experience", 1),
    ("I enjoyed every moment", 1),
    ("The story was amazing", 1),
    ("This product works perfectly", 1),
    ("The service was really good", 1),
    ("I am very happy with this", 1),
    ("This is a brilliant performance", 1),
    ("I hate this movie", 0),
    ("This film was terrible", 0),
    ("The acting was awful", 0),
    ("What a disappointing experience", 0),
    ("I disliked every moment", 0),
    ("The story was boring", 0),
    ("This product is broken", 0),
    ("The service was really bad", 0),
    ("I am very unhappy with this", 0),
    ("This is a poor performance", 0),
]


def tokenize(text: str) -> list[str]:
    """Basic word tokenization for English text."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


# Make a vocabulary: each word is converted into an integer index.
vocab = {"<PAD>": 0, "<UNK>": 1}
for sentence, _ in SAMPLES:
    for word in tokenize(sentence):
        if word not in vocab:
            vocab[word] = len(vocab)

PAD_ID = vocab["<PAD>"]
UNK_ID = vocab["<UNK>"]


def encode(text: str) -> list[int]:
    ids = [vocab.get(word, UNK_ID) for word in tokenize(text)]
    return ids or [UNK_ID]


def collate_batch(batch):
    """Pad variable-length sentences so they can be processed together."""
    sequences = [torch.tensor(encode(text), dtype=torch.long) for text, _ in batch]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    return padded, lengths, labels


loader = DataLoader(SAMPLES, batch_size=4, shuffle=True, collate_fn=collate_batch)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, model_type: str, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        if model_type == "RNN":
            self.recurrent = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif model_type == "LSTM":
            self.recurrent = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("model_type must be RNN or LSTM")

        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids, lengths):
        # (batch, tokens) -> (batch, tokens, embedding dimension)
        embeddings = self.embedding(input_ids)

        # Both RNN and LSTM return outputs at every word position.
        outputs, _ = self.recurrent(embeddings)

        # Select the last real word, ignoring padding at the end.
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        last_positions = lengths.to(input_ids.device) - 1
        last_hidden = outputs[rows, last_positions]

        # Return two raw scores (logits): Negative and Positive.
        return self.fc(last_hidden)


def train_model(model_type: str, epochs=60):
    model = TextClassifier(vocab_size=len(vocab), model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining {model_type}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, lengths, labels in loader:
            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 15 == 0:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss / len(loader):.4f}")

    return model


@torch.no_grad()
def training_accuracy(model) -> float:
    """Check fit on training examples (not unseen-data accuracy)."""
    model.eval()
    correct = 0
    for text, label in SAMPLES:
        ids = torch.tensor([encode(text)], dtype=torch.long)
        lengths = torch.tensor([ids.size(1)])
        predicted = model(ids, lengths).argmax(dim=1).item()
        correct += int(predicted == label)
    return correct / len(SAMPLES)


@torch.no_grad()
def predict(model, sentence: str) -> tuple[str, float]:
    model.eval()
    ids = torch.tensor([encode(sentence)], dtype=torch.long)
    lengths = torch.tensor([ids.size(1)])
    logits = model(ids, lengths)
    probabilities = torch.softmax(logits, dim=1)[0]
    label_id = int(torch.argmax(probabilities).item())
    label = "Positive" if label_id == 1 else "Negative"
    return label, float(probabilities[label_id].item())


if __name__ == "__main__":
    rnn_model = train_model("RNN")
    lstm_model = train_model("LSTM")

    print("\nTraining accuracy (not test accuracy):")
    print(f"RNN : {training_accuracy(rnn_model):.1%}")
    print(f"LSTM: {training_accuracy(lstm_model):.1%}")

    test_sentences = [
        "I really love this film",
        "The product is awful",
        "The performance was amazing",
        "This was a disappointing movie",
    ]

    print("\nSentiment predictions on example sentences:")
    for sentence in test_sentences:
        rnn_label, rnn_conf = predict(rnn_model, sentence)
        lstm_label, lstm_conf = predict(lstm_model, sentence)
        print(f"\nText: {sentence}")
        print(f"RNN : {rnn_label} (model probability: {rnn_conf:.2%})")
        print(f"LSTM: {lstm_label} (model probability: {lstm_conf:.2%})")
