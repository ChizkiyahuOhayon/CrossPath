# Figure captions

## CrossPath framework

**CrossPath exposes off-diagonal compatibility between frozen CIR endpoints.**
A composed query and gallery are encoded by two endpoints (left), all four
$q_i^\top g_j$ paths form a compatibility matrix (center), and zero-parameter
cross-mean fusion produces one ranking (right). In the shown FashionGen case,
the target moves from rank 9 to rank 1.

## Compatibility matrix

**Directional compatibility among three FashionIQ endpoints.** Each cell reports
the average recall obtained by pairing the row query encoder with the column
gallery encoder. The strongest off-diagonal paths use MCoT queries with DQU
galleries, motivating CrossPath fusion.

## Positive module ablation

**Each component yields a positive cumulative gain.** FashionGen reports the sum
of R@1, R@5, and R@10; FashionIQ reports R@10+R@50 averaged over dress, shirt,
and toptee.

## Retrieval examples

**Qualitative retrieval results on FashionGen and FashionIQ.** Reference and
target images are followed by the top five results from the strongest endpoint
and CrossPath. Green borders identify the target. Cases are selected
deterministically as the largest base-miss/CrossPath-hit rank improvements,
without manual visual screening.
