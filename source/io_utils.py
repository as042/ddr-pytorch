import json
import torch


def save_model(model, scaler, path, txt_format=False):
    if txt_format:
        state_dict = model.state_dict()

        readable_state = {}

        for k, v in state_dict.items():
            readable_state[k] = v.detach().cpu().tolist()

        debug_data = {
            "model_state_dict": readable_state,
            "scaler": {
                "mean_": scaler.mean_.tolist(),
                "scale_": scaler.scale_.tolist(),
                "var_": scaler.var_.tolist()
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2)

    else:
        package = {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
        }

        torch.save(package, path)

    print(f"Saved model to {path}")


def load_model(model_class, path):
    package = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    scaler = package["scaler"]

    model = model_class(scaler.n_features_in_)

    model.load_state_dict(package["model_state_dict"])

    print(f"Loaded model from {path}")

    return model, scaler