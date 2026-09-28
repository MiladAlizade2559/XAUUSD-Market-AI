from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )
