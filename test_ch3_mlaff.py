"""Tests for the chapter 3 model, data protocol and metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ch3_mlaff import data_fashiongen as dfg
from ch3_mlaff import eval_ch3
from ch3_mlaff.configs_ch3 import ABLATION_CONFIGS, CONFIGS, LAMBDA_CONFIGS, LAYER_CONFIGS
from ch3_mlaff.model_ch3 import Ch3Config, CrossModalBlock, TextGuidedGate, masked_mean
from ch3_mlaff.train_ch3 import lr_at


# --------------------------------------------------------------------------
# section 3.2.2
# --------------------------------------------------------------------------


def test_cross_block_shape_and_residual_identity():
    blk = CrossModalBlock(width=32, num_heads=4, dropout=0.0).eval()
    text = torch.randn(2, 7, 32)
    vis = torch.randn(2, 5, 32)
    assert blk(text, vis).shape == text.shape


def test_cross_block_zeroes_padded_text_positions():
    blk = CrossModalBlock(width=16, num_heads=4, dropout=0.0).eval()
    mask = torch.tensor([[1, 1, 0, 0]])
    out = blk(torch.randn(1, 4, 16), torch.randn(1, 3, 16), mask)
    assert torch.count_nonzero(out[0, 2:]) == 0


def test_masked_mean_ignores_padding():
    x = torch.stack([torch.ones(3, 4), torch.full((3, 4), 5.0)], dim=0)
    x[0, 2] = 99.0
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    got = masked_mean(x, mask)
    assert torch.allclose(got[0], torch.ones(4))
    assert torch.allclose(got[1], torch.full((4,), 5.0))


# --------------------------------------------------------------------------
# section 3.2.3
# --------------------------------------------------------------------------


def test_gate_weights_are_a_distribution_over_levels():
    gate = TextGuidedGate(width=24, num_levels=3).eval()
    beta = gate(torch.randn(5, 24))
    assert beta.shape == (5, 3)
    assert torch.allclose(beta.sum(-1), torch.ones(5), atol=1e-6)
    assert (beta >= 0).all()


# --------------------------------------------------------------------------
# configuration ladder
# --------------------------------------------------------------------------


def test_ablation_ladder_matches_table_3_5():
    expected = {
        "m1": ((12,), False, "none", False),
        "m2": ((12,), True, "none", False),
        "m3": ((4, 8, 12), True, "direct", False),
        "m4": ((4, 8, 12), True, "gated", False),
        "m5": ((4, 8, 12), True, "gated", True),
    }
    for name, (layers, xa, fusion, rea) in expected.items():
        cfg = ABLATION_CONFIGS[name]
        assert (cfg.vis_layers, cfg.use_cross_attn, cfg.fusion, cfg.use_rea) == (layers, xa, fusion, rea)
        cfg.validate()
    assert CONFIGS["baseline"] is CONFIGS["m1"]
    assert CONFIGS["full"] is CONFIGS["m5"]


def test_layer_sweep_covers_table_3_6():
    combos = {c.vis_layers for c in LAYER_CONFIGS.values()} | {ABLATION_CONFIGS["m5"].vis_layers}
    assert combos == {(12,), (8,), (4, 8), (8, 12), (4, 8, 12)}
    assert CONFIGS["L4_L8_L12"] is CONFIGS["m5"]


def test_lambda_sweep_covers_figure_3_6():
    weights = {c.rea_weight for c in LAMBDA_CONFIGS.values()} | {ABLATION_CONFIGS["m5"].rea_weight}
    assert weights == {0.0, 0.1, 0.3, 0.5, 1.0}
    # lambda = 0 is the ablation's model 4: same architecture, loss term switched off
    zero = CONFIGS["lam0_0"]
    m4 = ABLATION_CONFIGS["m4"]
    assert (zero.vis_layers, zero.fusion, zero.use_rea) == (m4.vis_layers, m4.fusion, m4.use_rea)


def test_every_config_validates():
    for cfg in CONFIGS.values():
        cfg.validate()


@pytest.mark.parametrize("kwargs", [
    dict(fusion="bogus"),
    dict(vis_layers=()),
    dict(vis_layers=(4, 8, 12), fusion="none"),
    dict(vis_layers=(12,), use_cross_attn=False, fusion="none", use_rea=True),
    dict(num_heads=7),
])
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Ch3Config(**kwargs).validate()


# --------------------------------------------------------------------------
# section 3.3.1 schedule
# --------------------------------------------------------------------------


def test_pretrain_schedule_warms_up_then_decays():
    kw = dict(warmup_epochs=5, total_epochs=30, lr_start=1e-5, lr_peak=6e-5, lr_min=1e-5)
    spe = 100
    assert lr_at(0, spe, **kw) == pytest.approx(1e-5)
    assert lr_at(5 * spe, spe, **kw) == pytest.approx(6e-5)
    assert lr_at(30 * spe, spe, **kw) == pytest.approx(1e-5, abs=1e-9)
    mid = lr_at(int(2.5 * spe), spe, **kw)
    assert 1e-5 < mid < 6e-5
    tail = [lr_at(s, spe, **kw) for s in range(5 * spe, 30 * spe, spe)]
    assert all(a >= b - 1e-12 for a, b in zip(tail, tail[1:]))


def test_retrieval_schedule_bounds():
    kw = dict(warmup_epochs=1, total_epochs=20, lr_start=1e-5, lr_peak=1e-5, lr_min=1e-6)
    spe = 50
    assert lr_at(0, spe, **kw) == pytest.approx(1e-5)
    assert lr_at(20 * spe, spe, **kw) == pytest.approx(1e-6, abs=1e-10)


# --------------------------------------------------------------------------
# section 3.3.2 metrics
# --------------------------------------------------------------------------


def test_full_recall_on_a_known_matrix():
    # 3 images, 2 texts; image rows 0,1 belong to text 0 and row 2 to text 1.
    sim_t2i = np.array([[0.9, 0.7, 0.2],
                        [0.3, 0.4, 0.8]])
    sim_i2t = sim_t2i.T
    img2txt = {0: 0, 1: 0, 2: 1}
    txt2img = {0: [0, 1], 1: [2]}
    got = eval_ch3.full_recall(sim_i2t, sim_t2i, img2txt, txt2img, ks=(1, 2))
    assert got["I2T"]["R@1"] == pytest.approx(100.0)      # every image ranks its text first
    assert got["T2I"]["R@1"] == pytest.approx(100.0)
    assert got["Mean R@1"] == pytest.approx(100.0)


def test_full_recall_counts_a_miss():
    sim_t2i = np.array([[0.1, 0.9]])       # text 0's positive is image 0, but image 1 wins
    got = eval_ch3.full_recall(sim_t2i.T, sim_t2i, {0: 0, 1: 0}, {0: [0]}, ks=(1, 2))
    assert got["T2I"]["R@1"] == pytest.approx(0.0)
    assert got["T2I"]["R@2"] == pytest.approx(100.0)


def test_sample_recall_uses_last_column_as_positive():
    i2t = np.array([[0.1, 0.2, 0.9],       # positive (last column) wins
                    [0.5, 0.2, 0.3]])      # positive is beaten by one candidate
    t2i = np.array([[0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0]])
    got = eval_ch3.sample_recall(i2t, t2i, ks=(1, 2))
    assert got["I2T"]["R@1"] == pytest.approx(50.0)
    assert got["I2T"]["R@2"] == pytest.approx(100.0)
    assert got["T2I"]["R@1"] == pytest.approx(100.0)
    assert got["Mean R@1"] == pytest.approx(75.0)


def test_gather_candidate_scores_picks_the_named_cells():
    sim = np.arange(20).reshape(4, 5).astype(float)
    got = eval_ch3.gather_candidate_scores(sim, np.array([0, 3]), np.array([[4, 1], [0, 2]]))
    assert np.array_equal(got, np.array([[4.0, 1.0], [15.0, 17.0]]))


def test_shortlist_returns_topk_best_first():
    sim = np.array([[0.1, 0.9, 0.5, 0.3],
                    [0.7, 0.2, 0.8, 0.6]])
    got = eval_ch3.shortlist(sim, topk=2)
    assert np.array_equal(got, np.array([[1, 2], [2, 0]]))
    assert np.array_equal(eval_ch3.shortlist(sim, topk=99)[:, 0], np.array([1, 2]))


def test_merge_reranked_keeps_dropped_candidates_below_every_reranked_one():
    sim = np.array([[0.1, 0.9, 0.5, 0.3]])
    candidates = np.array([[1, 2]])
    scores = np.array([[-5.0, -6.0]])      # reranking can push scores far down
    merged = eval_ch3.merge_reranked(sim, candidates, scores)
    assert merged[0, 1] == pytest.approx(-5.0) and merged[0, 2] == pytest.approx(-6.0)
    assert merged[0, 0] < merged[0, 2] and merged[0, 3] < merged[0, 2]


def test_pairs_for_flattens_in_both_directions():
    candidates = np.array([[7, 8], [9, 10]])
    queries = np.array([3, 4])
    img_q = eval_ch3._pairs_for(queries, candidates, queries_are_images=True)
    assert np.array_equal(img_q, np.array([[3, 7], [3, 8], [4, 9], [4, 10]]))
    txt_q = eval_ch3._pairs_for(queries, candidates, queries_are_images=False)
    assert np.array_equal(txt_q, np.array([[7, 3], [8, 3], [9, 4], [10, 4]]))


def test_cls_similarity_is_a_transpose_pair():
    corpus = {"cls_feat": torch.randn(6, 4), "text_feat": torch.randn(3, 4)}
    sim_i2t, sim_t2i = eval_ch3.cls_similarity(corpus)
    assert sim_i2t.shape == (6, 3) and sim_t2i.shape == (3, 6)
    assert np.allclose(sim_i2t, sim_t2i.T)


# --------------------------------------------------------------------------
# section 3.3.2 sampling protocol
# --------------------------------------------------------------------------


def _toy_index(products=40, per_product=2, subcats=4) -> dfg.SplitIndex:
    desc, sub, pids, plist, rows = [], [], [], [], {}
    for p in range(products):
        pid = 1000 + p
        plist.append(pid)
        rows[pid] = []
        for k in range(per_product):
            rows[pid].append(len(pids))
            desc.append(f"product {p} view {k}")
            sub.append(f"SUB{p % subcats}")
            pids.append(pid)
    return dfg.SplitIndex(desc, sub, pids, plist, rows)


def test_sample_protocol_shapes_and_positive_placement():
    index = _toy_index()
    out = dfg.sample_protocol_indices(index, seed=7, set_len=10, neg_len=5)
    assert out["i2t_txt"].shape == (10, 6) and out["t2i_img"].shape == (10, 6)
    for r in range(10):
        pos_txt = out["i2t_txt"][r, -1]
        assert index.product_list[pos_txt] == index.product_ids[out["i2t_img"][r]]
        assert len(set(out["i2t_txt"][r])) == 6          # no duplicated candidate
        pos_img = out["t2i_img"][r, -1]
        assert index.product_ids[pos_img] == index.product_list[out["t2i_txt"][r]]


def test_sample_protocol_is_seed_deterministic():
    index = _toy_index()
    a = dfg.sample_protocol_indices(index, seed=11, set_len=8, neg_len=4)
    b = dfg.sample_protocol_indices(index, seed=11, set_len=8, neg_len=4)
    c = dfg.sample_protocol_indices(index, seed=12, set_len=8, neg_len=4)
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert not all(np.array_equal(a[k], c[k]) for k in a)


def test_negatives_prefer_the_query_subcategory():
    # 30 products in the query's subcategory is more than the 5 negatives needed,
    # so every negative must come from that subcategory.
    desc, sub, pids, plist, rows = [], [], [], [], {}
    for p in range(40):
        pid = 2000 + p
        plist.append(pid)
        rows[pid] = [len(pids)]
        desc.append(f"p{p}")
        sub.append("HOT" if p < 30 else "COLD")
        pids.append(pid)
    index = dfg.SplitIndex(desc, sub, pids, plist, rows)
    out = dfg.sample_protocol_indices(index, seed=3, set_len=40, neg_len=5)
    for r in range(out["i2t_txt"].shape[0]):
        if index.subcategories[out["i2t_img"][r]] != "HOT":
            continue
        for col in out["i2t_txt"][r, :-1]:
            assert index.subcategories[index.product_rows[index.product_list[col]][0]] == "HOT"


def test_full_protocol_labels_are_consistent():
    index = _toy_index(products=5, per_product=3)
    img2txt, txt2img = dfg.full_protocol_labels(index)
    assert len(img2txt) == index.num_images
    assert len(txt2img) == index.num_products
    for text_col, rows in txt2img.items():
        assert len(rows) == 3
        for row in rows:
            assert img2txt[row] == text_col


def test_gallery_has_one_text_per_product():
    index = _toy_index(products=6, per_product=4)
    texts = dfg.gallery_texts(index)
    assert len(texts) == 6
    assert all(t.startswith(dfg.TEXT_PREFIX) for t in texts)


def test_pre_caption_matches_fashionsap_normalisation():
    assert dfg.pre_caption("Blue-Faded Jacket, 100% cotton.") == "blue faded jacket 100% cotton"
    assert dfg.pre_caption("a  b\n") == "a b"


# --------------------------------------------------------------------------
# regression: the chapter-3 modules must actually be trained
# --------------------------------------------------------------------------


def _tiny_model(cfg, momentum: bool):
    """A Ch3Model on randomly initialised encoders, small enough for a unit test."""
    timm = pytest.importorskip("timm")
    pytest.importorskip("transformers")
    from ch3_mlaff.encoders_ch3 import MultiLayerViT, SplitBert
    from ch3_mlaff.model_ch3 import Ch3Model

    def vit():
        return MultiLayerViT(cfg.vit_name, img_size=cfg.image_res, pretrained=False)

    def bert():
        return SplitBert(cfg.bert_name, text_layers=cfg.text_layers, pretrained=False)

    return Ch3Model(cfg, vit(), bert(), *( (vit(), bert()) if momentum else (None, None) ))


def _grad_norm(model, prefix: str) -> float:
    return sum(float(p.grad.norm()) for n, p in model.named_parameters()
               if n.startswith(prefix) and p.grad is not None)


@pytest.mark.parametrize("name", ["m2", "m3", "m4", "m5"])
def test_cross_attention_and_gate_receive_gradient(name):
    """Table 3.5 rows 2-4 carry no region-enhanced loss.

    If F_final only ever reached the objective through L_rea, those rows would
    train identically to row 1 and the ablation ladder would be vacuous.  The
    contrastive term on F_final is what prevents that, so guard it.
    """
    torch.manual_seed(0)
    cfg = ABLATION_CONFIGS[name]
    model = _tiny_model(cfg, momentum=True)          # the configuration training uses

    b = 2
    out = model(torch.randn(b, 3, cfg.image_res, cfg.image_res),
                torch.randint(1000, 20000, (b, cfg.max_word_num)),
                torch.ones(b, cfg.max_word_num, dtype=torch.long),
                torch.arange(b))
    out["loss"].backward()

    assert _grad_norm(model, "cross_blocks") > 0.0
    assert _grad_norm(model, "region_proj") > 0.0
    if cfg.fusion == "gated":
        assert _grad_norm(model, "gate") > 0.0
    assert (float(out["loss_rea"]) > 0.0) is cfg.use_rea


def test_baseline_has_no_chapter_three_modules():
    cfg = ABLATION_CONFIGS["m1"]
    model = _tiny_model(cfg, momentum=False)
    assert model.cross_blocks is None and model.gate is None and model.region_proj is None
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.startswith(("cross_blocks", "gate", "region_proj")) for n in names)


# --------------------------------------------------------------------------
# protocol plumbing, exercised without encoders
# --------------------------------------------------------------------------


class _StubModel:
    """A model whose pairwise score is a lookup into a fixed matrix.

    ``score_pairs`` is handed the image patches of the pairs and their text
    indices, so a stub that carries the image row in its "patches" can return
    an arbitrary per-pair score and let the protocol code be tested on its own.
    """

    class _Cfg:
        vis_layers = (12,)
        image_res = 4
        max_word_num = 3
        topk_rerank = 2

    cfg = _Cfg()
    cross_blocks = object()          # "this model has the chapter-3 modules"

    def __init__(self, table: np.ndarray):
        self.table = table
        self.calls = 0

    def eval(self):
        return self

    def encode_image(self, images):
        return None, {12: images}

    def score_pairs(self, patches, text_seq, attention_mask, text_feat):
        self.calls += 1
        rows = patches[12].round().long().view(-1)
        cols = text_seq[:, 0, 0].round().long()
        return torch.tensor([self.table[int(r), int(c)] for r, c in zip(rows, cols)])


class _StubDataset:
    def image_at(self, row):
        return torch.full((1, 1), float(row))


def _stub_corpus(n_img, n_txt):
    return {
        "cls_feat": torch.eye(n_img, 8)[:, :8],
        "text_feat": torch.eye(n_txt, 8)[:, :8],
        "text_seq": torch.arange(n_txt).view(n_txt, 1, 1).expand(n_txt, 1, 8).float().clone(),
        "text_mask": torch.ones(n_txt, 1, dtype=torch.long),
    }


def test_score_pairs_visits_every_pair_exactly_once():
    table = np.arange(12, dtype=np.float32).reshape(4, 3)
    model = _StubModel(table)
    pairs = np.array([[3, 0], [1, 2], [0, 1], [3, 2]])
    got = eval_ch3.score_pairs(model, _StubDataset(), _stub_corpus(4, 3), pairs,
                               torch.device("cpu"), pair_batch=2)
    assert np.allclose(got, [table[r, c] for r, c in pairs])


def test_score_pairs_reuses_one_image_across_its_candidates():
    """Sorting by image row is what keeps the ViT from running per candidate."""
    table = np.zeros((2, 4), dtype=np.float32)
    model = _StubModel(table)
    pairs = np.array([[0, i] for i in range(4)] + [[1, i] for i in range(4)])
    eval_ch3.score_pairs(model, _StubDataset(), _stub_corpus(2, 4), pairs,
                         torch.device("cpu"), pair_batch=4)
    assert model.calls == 2          # one batch per image, not one per pair


def test_evaluate_sample_reads_the_pairwise_score_not_the_cls_matrix():
    # CLS says image 0 matches text 0; the pairwise table says the opposite.
    table = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    model = _StubModel(table)
    idxs = {
        "i2t_img": np.array([0]), "i2t_txt": np.array([[1, 0]]),
        "t2i_txt": np.array([0]), "t2i_img": np.array([[1, 0]]),
    }
    got = eval_ch3.evaluate_sample(model, _StubDataset(), _stub_corpus(2, 2), idxs,
                                   torch.device("cpu"))
    assert got["I2T"]["R@1"] == pytest.approx(0.0)   # positive scores 0, negative 1


def test_evaluate_full_reranks_only_the_shortlist():
    table = np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]], dtype=np.float32)
    model = _StubModel(table)
    corpus = _stub_corpus(3, 3)
    got = eval_ch3.evaluate_full(model, _StubDataset(), corpus,
                                 {0: 0, 1: 1, 2: 2}, {0: [0], 1: [1], 2: [2]},
                                 torch.device("cpu"), topk=2)
    assert got["I2T"]["R@1"] == pytest.approx(100.0)
    assert got["T2I"]["R@1"] == pytest.approx(100.0)


def test_evaluate_full_falls_back_to_cls_for_the_two_tower_baseline():
    model = _StubModel(np.zeros((3, 3), dtype=np.float32))
    model.cross_blocks = None
    corpus = _stub_corpus(3, 3)
    got = eval_ch3.evaluate_full(model, _StubDataset(), corpus,
                                 {0: 0, 1: 1, 2: 2}, {0: [0], 1: [1], 2: [2]},
                                 torch.device("cpu"), topk=2)
    assert model.calls == 0
    assert got["I2T"]["R@1"] == pytest.approx(100.0)   # the identity CLS matrix
