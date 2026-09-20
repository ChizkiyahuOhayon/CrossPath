"""Chapter 3: multi-level adaptive feature fusion for fashion image-text retrieval (MLAFF).

Implements the model specified in thesis section 3.2:
  * multi-level ViT-B/16 patch features from blocks 4 / 8 / 12   (3.2.1)
  * hierarchical cross-modal cross-attention, text tokens as Query (3.2.2, Eq. 3.1-3.7)
  * text-guided gated fusion over the three levels           (3.2.3, Eq. 3.8-3.10)
  * region-enhanced alignment loss                           (3.2.4, Eq. 3.11-3.13)
"""
