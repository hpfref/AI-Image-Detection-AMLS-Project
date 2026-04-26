# Dockerfile for the AMLS 2026 graded submission.
#
# Constraints (PDF §Submission Guidelines):
#   - CPU-only; no internet access at runtime.
#   - Final image must be <= 4 GB.
#   - The built image is NOT included in the zip — only this Dockerfile +
#     requirements.txt + source.
#   - At runtime, /workspace/solution/data is a read-only mount; write only
#     under /workspace/solution/artifacts.
#
# Body below is verbatim from PDF Appendix A.

# --- BEGIN: copied from PDF Appendix A ---
FROM python:3.11-slim

WORKDIR /workspace/solution

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/solution/requirements.txt
RUN pip install --no-cache-dir -r /workspace/solution/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

COPY . /workspace/solution
# --- END: copied from PDF Appendix A ---
