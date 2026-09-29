# Builds and serves the Chapter-5 retrieval demo (system/). Build context is
# the repo root because crosspath/pipeline.py imports weave_crosspath.py and
# weave_crosspath_gate.py from here — see system/README.md "分层".
#
# No FashionGen data is available inside the image, so db/build_db.py always
# falls back to the repo's own synthetic placeholder catalog
# (system/data/demo_placeholder/, 15 hand-drawn garments, no licensing issue).
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch/torchvision — the default PyPI wheels pull in CUDA and are
# several GB heavier than this demo needs.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY system/requirements-docker.txt system/requirements-docker.txt
RUN pip install --no-cache-dir -r system/requirements-docker.txt

COPY weave_crosspath.py weave_crosspath_gate.py ./
COPY system/app.py system/app.py
COPY system/crosspath system/crosspath
COPY system/db system/db
COPY system/templates system/templates
COPY system/static system/static
COPY system/tools/train_endpoints.py system/tools/train_gate.py system/tools/
COPY system/data/demo_placeholder system/data/demo_placeholder

WORKDIR /app/system

# Build the catalog and train the two endpoint heads + gate at image-build
# time (also warms the CLIP pretrained-weight cache into the image, so the
# running container never needs outbound network access).
RUN mkdir -p data/uploads && \
    python db/build_db.py && \
    python tools/train_endpoints.py && \
    python tools/train_gate.py

ENV HOST=0.0.0.0
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
