import torch
import pickle

cifar10_dataset_dir = "datasets/cifar-10-batches-py/"

cifar10_filename_list = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
]

poison_delta_ckpt = 'pretrained_poisons/cifar10_res18_simclr_cps.pth'

def load_dataset_batch(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data # batch_label, labels, data, filenames

def load_poison_delta(path: str):
    return torch.load(path, map_location='cpu') # (50000, 3, 32, 32)

def main():
    pass


if __name__ == "__main__":
    main()
