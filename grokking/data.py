from math import ceil
import torch


DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x+y),
    "x-y": lambda x, y, _: (x, y, x-y),
    **DIVISION_MODULO_OPERATIONS,
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}

def operation_mod_p_data(operation: str, p: int, eq_token: int, op_token: int):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p)
    y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    x, y, labels = ALL_OPERATIONS[operation](x, y, p)

    inputs = torch.stack([x, op, y, eq], dim=1)

    return inputs, labels

def operation_mod_p_data_noisy(operation: str, p: int, eq_token: int, op_token: int, noise_level: float, noise_cols):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p)
    y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    x, y, labels = ALL_OPERATIONS[operation](x, y, p)

    inputs = torch.stack([x, op, y, eq], dim=1).float()

    return inputs, labels


def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):


    inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
    # inputs, labels = operation_mod_p_data_noisy(operation, prime, prime, prime+1, noise_level = noise_level, noise_cols = noise_cols)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    # WHY

    val_size = int(len(dataset) * 0.2)
    train_size = int( (len(dataset) - val_size) * training_fraction)
    remain_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size, remain_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, train_size, val_size
