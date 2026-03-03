# ── Stage 1: Build ──────────────────────────────────────
FROM node:22-alpine AS build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

COPY frontend/ .

ARG GROQ_API_KEY
ENV GROQ_API_KEY=${GROQ_API_KEY}

RUN npm run build

# ── Stage 2: Serve with nginx ───────────────────────────
FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/dist /usr/share/nginx/html

# GLB models are NOT baked in — mounted as a volume at /usr/share/nginx/models

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget -qO- http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
