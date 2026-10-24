# LeafLens web app. CI builds and smoke-tests this image on every push; Hugging Face Spaces
# builds it on every deploy.
FROM python:3.10-slim

# Hugging Face Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2
WORKDIR $HOME/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

COPY --chown=user app.py .
COPY --chown=user plant_disease plant_disease
COPY --chown=user templates templates
COPY --chown=user static static
COPY --chown=user models models

EXPOSE 7860
# One worker keeps a single copy of the model in memory; its threads serve requests in parallel.
# PORT lets other hosts (Render, Cloud Run, ...) choose the port; Spaces use 7860.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 'app:create_app()'"]
