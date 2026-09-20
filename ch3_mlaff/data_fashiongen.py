"""FashionGen data pipeline for chapter 3.

The h5 layout and the evaluation protocol follow the FashionSAP release that
the chapter's baseline reproduces (``hssip/FashionSAP``): text is
``"the image description is " + pre_caption(input_description)`` truncated to
``max_word_num`` tokens, the text gallery holds one description per product,
and the Sample protocol scores 1 positive against 100 negatives drawn first
from the query's own subcategory.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_PUNCT = re.compile(r"([,.'!\"()*#:;~])")
_SPACES = re.compile(r"\s{2,}")

TEXT_PREFIX = "the image description is "


def pre_caption(caption: str) -> str:
    """Normalisation used by the FashionSAP dataloader."""
    caption = _PUNCT.sub("", caption.lower())
    caption = caption.replace("-", " ").replace("/", " ")
    caption = caption.replace("<person>", "person").replace("<br>", " ")
    return _SPACES.sub(" ", caption).rstrip("\n").strip(" ")


def _decode(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode("iso-8859-1")
    return str(raw)


# --------------------------------------------------------------------------
# split index, built once from the h5 file and cached as json
# --------------------------------------------------------------------------


@dataclass
class SplitIndex:
    """Per-split metadata extracted from the h5 file."""

    descriptions: List[str]          # one entry per image row
    subcategories: List[str]         # one entry per image row
    product_ids: List[int]           # one entry per image row
    product_list: List[int]          # unique product ids, in file order
    product_rows: Dict[int, List[int]]   # product id -> image row indices

    @property
    def num_images(self) -> int:
        return len(self.product_ids)

    @property
    def num_products(self) -> int:
        return len(self.product_list)

    def to_json(self, path: str) -> None:
        payload = {
            "descriptions": self.descriptions,
            "subcategories": self.subcategories,
            "product_ids": self.product_ids,
            "product_list": self.product_list,
            "product_rows": {str(k): v for k, v in self.product_rows.items()},
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def from_json(cls, path: str) -> "SplitIndex":
        with open(path, encoding="utf-8") as fh:
            p = json.load(fh)
        return cls(
            descriptions=p["descriptions"],
            subcategories=p["subcategories"],
            product_ids=p["product_ids"],
            product_list=p["product_list"],
            product_rows={int(k): v for k, v in p["product_rows"].items()},
        )

    @classmethod
    def build(cls, h5_path: str) -> "SplitIndex":
        import h5py

        with h5py.File(h5_path, "r") as fh:
            desc = [_decode(x) for x in fh["input_description"][:, 0]]
            subcat = [_decode(x) for x in fh["input_subcategory"][:, 0]]
            pids = [int(x) for x in fh["input_productID"][:, 0]]

        product_list: List[int] = []
        product_rows: Dict[int, List[int]] = {}
        for row, pid in enumerate(pids):
            if pid not in product_rows:
                product_rows[pid] = []
                product_list.append(pid)
            product_rows[pid].append(row)
        return cls(desc, subcat, pids, product_list, product_rows)

    @classmethod
    def load_or_build(cls, h5_path: str, cache_path: str) -> "SplitIndex":
        if os.path.exists(cache_path):
            return cls.from_json(cache_path)
        index = cls.build(h5_path)
        index.to_json(cache_path)
        return index


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


class _FashionGenBase(Dataset):
    """Shared h5 handling.  The file handle is opened lazily per worker."""

    def __init__(self, h5_path: str, index: SplitIndex, tokenizer, transform,
                 max_words: int = 180):
        self.h5_path = h5_path
        self.index = index
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_words = max_words
        self._fh = None

    def _images(self):
        if self._fh is None:
            import h5py

            self._fh = h5py.File(self.h5_path, "r")
        return self._fh["input_image"]

    def image_at(self, row: int) -> torch.Tensor:
        from PIL import Image

        return self.transform(Image.fromarray(self._images()[row]))

    def text_at(self, row: int) -> str:
        return TEXT_PREFIX + pre_caption(self.index.descriptions[row])

    def tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """CLS + wordpieces, right-padded to ``max_words`` (FashionSAP layout)."""
        tokens = ["[CLS]"] + self.tokenizer.tokenize(text)
        tokens = tokens[: self.max_words]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        pad = self.max_words - len(ids)
        mask = [1] * len(ids) + [0] * pad
        ids = ids + [0] * pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None          # h5py handles are not picklable
        return state


class FashionGenPretrain(_FashionGenBase):
    """Stage-1 domain pre-training: every image row is one sample.

    Returns ``(image, input_ids, attention_mask, mlm_ids, mlm_labels, product_id)``.
    """

    def __init__(self, *args, mlm_probability: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlm_probability = mlm_probability
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.vocab_size = self.tokenizer.vocab_size
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.pid_to_label = {pid: i for i, pid in enumerate(self.index.product_list)}

    def __len__(self) -> int:
        return self.index.num_images

    def _mask_tokens(self, ids: torch.Tensor, mask: torch.Tensor):
        labels = ids.clone()
        prob = torch.full(ids.shape, self.mlm_probability)
        for i, tok in enumerate(ids.tolist()):
            if tok in self.special_ids or mask[i] == 0:
                prob[i] = 0.0
        selected = torch.bernoulli(prob).bool()
        labels[~selected] = -100
        out = ids.clone()
        replace = torch.bernoulli(torch.full(ids.shape, 0.8)).bool() & selected
        out[replace] = self.mask_id
        rand = torch.bernoulli(torch.full(ids.shape, 0.5)).bool() & selected & ~replace
        out[rand] = torch.randint(self.vocab_size, (int(rand.sum()),), dtype=torch.long)
        return out, labels

    def __getitem__(self, i: int):
        ids, mask = self.tokenize(self.text_at(i))
        mlm_ids, mlm_labels = self._mask_tokens(ids, mask)
        label = self.pid_to_label[self.index.product_ids[i]]
        return self.image_at(i), ids, mask, mlm_ids, mlm_labels, label


class FashionGenRetrievalTrain(_FashionGenBase):
    """Stage-2 retrieval fine-tuning: one sample per *product*.

    A random image of the product is drawn each epoch, matching the FashionSAP
    retrieval dataloader.
    """

    def __len__(self) -> int:
        return self.index.num_products

    def __getitem__(self, i: int):
        pid = self.index.product_list[i]
        row = random.choice(self.index.product_rows[pid])
        ids, mask = self.tokenize(self.text_at(row))
        return self.image_at(row), ids, mask, i


class FashionGenEvalImages(_FashionGenBase):
    """All validation images, in file order."""

    def __len__(self) -> int:
        return self.index.num_images

    def __getitem__(self, i: int):
        return self.image_at(i), i


# --------------------------------------------------------------------------
# evaluation protocol (section 3.3.2)
# --------------------------------------------------------------------------


def gallery_texts(index: SplitIndex) -> List[str]:
    """One description per product: the text gallery of the Full protocol."""
    return [TEXT_PREFIX + pre_caption(index.descriptions[index.product_rows[pid][0]])
            for pid in index.product_list]


def full_protocol_labels(index: SplitIndex) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """``(img2txt, txt2img)`` over the whole validation split."""
    txt2img = {i: list(index.product_rows[pid]) for i, pid in enumerate(index.product_list)}
    img2txt = {row: i for i, rows in txt2img.items() for row in rows}
    return img2txt, txt2img


def _sample_negative_products(index: SplitIndex, subcate_map: Dict[str, List[int]],
                              query_pid: int, query_subcat: str, rng: random.Random,
                              neg_len: int) -> List[int]:
    """100 negatives, own subcategory first, then random other subcategories."""
    others = [s for s in subcate_map if s != query_subcat]
    rng.shuffle(others)
    pool = [p for p in subcate_map.get(query_subcat, []) if p != query_pid]
    while len(pool) < neg_len:
        if not others:
            raise RuntimeError("not enough products to draw negatives from")
        pool.extend(subcate_map[others.pop()])
    return rng.sample(pool, neg_len)


def build_subcate_map(index: SplitIndex) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    seen: Dict[str, set] = {}
    for row, pid in enumerate(index.product_ids):
        sub = index.subcategories[row]
        bucket = seen.setdefault(sub, set())
        if pid not in bucket:
            bucket.add(pid)
            out.setdefault(sub, []).append(pid)
    return out


def sample_protocol_indices(index: SplitIndex, seed: int, set_len: int = 1000,
                            neg_len: int = 100) -> Dict[str, np.ndarray]:
    """Index arrays for the 101-candidate Sample protocol.

    Returns ``i2t_img``/``i2t_txt`` and ``t2i_txt``/``t2i_img``; in both
    candidate matrices the positive sits in the last column (column ``neg_len``).
    """
    rng = random.Random(seed)
    subcate_map = build_subcate_map(index)
    pid_to_col = {pid: i for i, pid in enumerate(index.product_list)}
    order = list(range(index.num_products))

    rng.shuffle(order)
    i2t_img, i2t_txt = [], []
    for col in order[:set_len]:
        pid = index.product_list[col]
        row = rng.choice(index.product_rows[pid])
        negs = _sample_negative_products(index, subcate_map, pid, index.subcategories[row], rng, neg_len)
        i2t_img.append(row)
        i2t_txt.append([pid_to_col[p] for p in negs] + [col])

    rng.shuffle(order)
    t2i_txt, t2i_img = [], []
    for col in order[:set_len]:
        pid = index.product_list[col]
        pos_row = rng.choice(index.product_rows[pid])
        negs = _sample_negative_products(index, subcate_map, pid, index.subcategories[pos_row], rng, neg_len)
        t2i_txt.append(col)
        t2i_img.append([rng.choice(index.product_rows[p]) for p in negs] + [pos_row])

    return {
        "i2t_img": np.asarray(i2t_img, dtype=np.int64),
        "i2t_txt": np.asarray(i2t_txt, dtype=np.int64),
        "t2i_txt": np.asarray(t2i_txt, dtype=np.int64),
        "t2i_img": np.asarray(t2i_img, dtype=np.int64),
    }


def build_transform(image_res: int, is_train: bool):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_res, scale=(0.5, 1.0),
                                         interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((image_res, image_res), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        norm,
    ])
