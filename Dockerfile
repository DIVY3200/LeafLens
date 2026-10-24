# LeafLens web app as deployed on Render. CI builds and smoke-tests this image on every push.
# It serves the TensorFlow Lite model on LiteRT, so it needs no TensorFlow and fits in 512 MB.
FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

COPY --chown=user app.py .
COPY --chown=user plant_disease plant_disease
COPY --chown=user templates templates
COPY --chown=user static static
COPY --chown=user models/leaflens_mobilenetv2.tflite models/leaflens_mobilenetv2.json models/

EXPOSE 10000
# Render sets PORT (10000 by default). One worker keeps a single copy of the model in memory;
# its threads serve requests in parallel.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 'app:create_app()'"]
