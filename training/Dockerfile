# TFLite export environment (the converter does not run on native Windows).
FROM python:3.12-slim
# CPU only torch, the default Linux wheel pulls several GB of CUDA libraries
RUN pip install --no-cache-dir "torch>=2.4,<2.15" --index-url https://download.pytorch.org/whl/cpu
COPY requirements-export.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-export.txt
WORKDIR /repo/training
