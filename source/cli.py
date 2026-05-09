import argparse

import pandas as pd

from dataset import DDRDataset
from model import DDRModel
from data_utils import clean_dataframe
from run import run_model
from train import train
from io_utils import save_model, load_model


def train_command(args):
    dataset = DDRDataset(args.dataset)
    model = DDRModel(input_dim=dataset.X.shape[1])

    if args.resume:
        model, _ = load_model(
            DDRModel,
            args.resume
        )
        path = f"{args.resume}_e{args.epochs}.pt"

    else:
        path = "output.pt"

    model = train(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    save_model(
        model,
        dataset.scaler,
        path=path
    )


def run_command(args):
    model, scaler = load_model(
        DDRModel,
        args.model
    )
    dataset = DDRDataset(args.dataset, scaler=scaler)

    sample = dataset[args.input_idx]
    X = sample[0].reshape(1, -1)

    print(f"y: {sample[1].item()}")

    y = run_model(model, X)

    print(f"Guess: {y}")


def clean_data_command(args):
    df = pd.read_csv(args.dataset_in, low_memory=False)

    cleaned_df = clean_dataframe(df)
    cleaned_df.to_csv(args.dataset_out, index=False)
    print(f"Saved cleaned dataset to {args.dataset_out}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("dataset")
    train_parser.add_argument(
        "--resume",
        default=None
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=80
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("model")
    run_parser.add_argument("dataset")
    run_parser.add_argument(
        "input_idx",
        type=int                    
    )

    clean_parser = subparsers.add_parser("clean_data")
    clean_parser.add_argument("dataset_in")
    clean_parser.add_argument("dataset_out")

    args = parser.parse_args()
    if args.command == "train":
        train_command(args)

    elif args.command == "run":
        run_command(args)

    elif args.command == "clean_data":
        clean_data_command(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()