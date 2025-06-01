FROM python:3.12.10-slim AS base

WORKDIR /app

# Stage 2: Install dependencies
FROM base AS builder
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final application image
FROM base AS runner
# Copy installed packages from the 'builder' stage's site-packages
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
# Copy your application code
COPY . .

EXPOSE 5000

CMD ["python", "server.py"]