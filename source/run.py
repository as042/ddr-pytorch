import torch


def run_model(model, input):
    with torch.no_grad():
        pred = model(input).item()

    return pred