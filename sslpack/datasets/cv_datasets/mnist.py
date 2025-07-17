from sslpack.datasets import Dataset
from torchvision.datasets import MNIST as MN
from torchvision import transforms
from sslpack.utils.data import TransformDataset, BasicDataset
from sslpack.utils.data import split_lb_ulb_balanced
from sslpack.utils.augmentation import RandAugment


class Mnist(Dataset):

    def __init__(
        self,
        lbls_per_class,
        ulbls_per_class=None,
        seed=None,
        return_ulbl_labels=False,
        crop_size=28,
        crop_ratio=1,
        data_dir="~/.sslpack/datasets/MNIST",
        download=True,
    ):

        mnist_tr = MN(root=data_dir, train=True, download=download)
        mnist_ts = MN(root=data_dir, train=False, download=download)

        X_tr, y_tr = mnist_tr.data.float() / 255, mnist_tr.targets
        X_ts, y_ts = mnist_ts.data.float() / 255, mnist_ts.targets

        X_tr = X_tr.unsqueeze(1)
        X_ts = X_ts.unsqueeze(1)

        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(crop_size),
                transforms.RandomCrop(
                    crop_size,
                    padding=int(crop_size * (1 - crop_ratio)),
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(X_tr.mean(), X_tr.std()),
            ]
        )
        self.strong_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(crop_size),
                transforms.RandomCrop(
                    crop_size,
                    padding=int(crop_size * (1 - crop_ratio)),
                    padding_mode="reflect",
                ),
                transforms.RandomHorizontalFlip(),
                RandAugment(3, 5, exclude_color_aug=True, bw=True),
                transforms.ToTensor(),
                transforms.Normalize(X_tr.mean(), X_tr.std()),
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(crop_size),
                transforms.Normalize(X_tr.mean(), X_tr.std()),
            ]
        )

        X_tr_lb, y_tr_lb, X_tr_ulb, y_tr_ulb = split_lb_ulb_balanced(
            X=X_tr,
            y=y_tr,
            lbls_per_class=lbls_per_class,
            ulbls_per_class=ulbls_per_class,
            seed=seed,
        )

        if not return_ulbl_labels:
            y_tr_ulb = None

        self.lbl_dataset = TransformDataset(
            X_tr_lb,
            y_tr_lb,
            transform=self.transform,
            weak_transform=self.weak_transform,
            strong_transform=self.strong_transform,
        )

        self.ulbl_dataset = TransformDataset(
            X_tr_ulb,
            y_tr_ulb,
            transform=self.transform,
            weak_transform=self.weak_transform,
            strong_transform=self.strong_transform,
        )

        X_ts, y_ts = X_ts.float(), y_ts.float()

        self.eval_dataset = BasicDataset(X_ts, y_ts, transform=self.transform)
