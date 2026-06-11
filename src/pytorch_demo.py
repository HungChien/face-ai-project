from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


def main():
    torch.manual_seed(42)

    output_dir = Path("outputs/reports")
    checkpoint_dir = Path("models/checkpoints")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples = 200
    input_dim = 10
    num_classes = 2
    num_epochs = 10
    learning_rate = 0.001

    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    x = x.to(device)
    y = y.to(device)

    model = SimpleClassifier(
        input_dim=input_dim,
        hidden_dim=32,
        output_dim=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logs = []
    logs.append("PyTorch Demo Training Result")
    logs.append("=" * 50)
    logs.append(f"Device: {device}")
    logs.append(f"Number of samples: {num_samples}")
    logs.append(f"Input dimension: {input_dim}")
    logs.append(f"Number of classes: {num_classes}")
    logs.append(f"Number of epochs: {num_epochs}")
    logs.append(f"Learning rate: {learning_rate}")
    logs.append("")

    for epoch in range(num_epochs):
        model.train()

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()

        log_line = (
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {loss.item():.4f} "
            f"Accuracy: {accuracy:.4f}"
        )

        print(log_line)
        logs.append(log_line)

    checkpoint_path = checkpoint_dir / "pytorch_demo.pth"
    torch.save(model.state_dict(), checkpoint_path)

    report_path = output_dir / "pytorch_demo_result.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))

    print("")
    print("PyTorch demo completed successfully.")
    print(f"Model checkpoint saved to: {checkpoint_path}")
    print(f"Training report saved to: {report_path}")


if __name__ == "__main__":
    main()