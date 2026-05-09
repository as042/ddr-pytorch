import torch
import torch.nn as nn

from torch.utils.data import DataLoader


def train(model, dataset, epochs, batch_size=32, lr=0.001):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.SmoothL1Loss()

    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in loader:
            preds = model(X_batch)

            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    return model