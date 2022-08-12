from comet_ml import Experiment

import tqdm
import torch
import argparse

from utils import *
from data.dataset import CustomDataset
from models.model import CustomModel


def main(args):
    config = get_config_yaml()
    visible_print(args.name)

    experiment = Experiment(
        api_key=config["experiment"]["api_key"],
        project_name=args.name,
        workspace=config["experiment"]["workspace"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visible_print("Device")
    print(device)

    train_ds = CustomDataset(
        data_dir=config["data"]["train"],
        tokenizer_path=config["data"]["tokenizer"],
        experiment=experiment,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)

    valid_ds = CustomDataset(data_dir=config["data"]["valid"], tokenizer_path=config["data"]["tokenizer"])
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    test_ds = CustomDataset(data_dir=config["data"]["test"], tokenizer_path=config["data"]["tokenizer"])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)
    visible_print("Dataset")
    print("Train dataset: %d files" % len(train_ds))
    print("Valid dataset: %d files" % len(valid_ds))
    print("Test dataset: %d files" % len(test_ds))

    experiment.log_parameters(config["params"])

    model = CustomModel()
    model.to(device)
    visible_print("Model architecture")
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    visible_print("Training")
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = 0.0
        train_acc = 0.0
        for (inputs, labels) in tqdm.tqdm(train_loader, desc="Epoch %d" % epoch):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, attn_weights = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss.item()
            train_acc += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

        train_loss /= len(train_ds)
        train_acc /= len(train_ds)

        experiment.log_metrics(
            {
                "train loss": train_loss,
                "train accuracy": train_acc,
            },
            epoch=epoch,
        )

        valid_loss = 0.0
        valid_acc = 0.0
        for (inputs, labels) in tqdm.tqdm(valid_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            valid_acc += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

        valid_loss /= len(valid_ds)
        valid_acc /= len(valid_ds)

        print(
            "Epoch: %d, train_loss: %.4f, train_acc: %.4f, valid_loss: %.4f, valid_acc: %.4f"
            % (epoch, train_loss, train_acc, valid_loss, valid_acc)
        )
        experiment.log_metrics(
            {
                "validation loss": valid_loss,
                "validation accuracy": valid_acc,
            },
            epoch=epoch,
        )

    visible_print("Testing")
    test_loss = 0.0
    test_acc = 0.0
    for (inputs, labels) in tqdm.tqdm(test_loader, desc="Testing"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, attn_weights = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        test_acc += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    test_loss /= len(test_ds)
    test_acc /= len(test_ds)

    print("test_loss: %.4f, test_acc: %.4f" % (test_loss, test_acc))
    experiment.log_metrics(
        {
            "test loss": test_loss,
            "test accuracy": test_acc,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="general")
    args = parser.parse_args()

    main(args)
