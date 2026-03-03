#!/usr/bin/env bash
set -euo pipefail

# RunPod setup script for F1 OmniSense
# Installs MongoDB + mongo-express directly (no Docker — RunPod IS a container).
# Stores MongoDB data on the persistent volume for survival across restarts.

echo "=== F1 OmniSense — RunPod Setup ==="

# ── Detect volume mount ───────────────────────────────────────
VOLUME_PATH=""
for p in /workspace /runpod-volume; do
    if [ -d "$p" ]; then
        VOLUME_PATH="$p"
        break
    fi
done

if [ -z "$VOLUME_PATH" ]; then
    echo "WARNING: No RunPod volume found. MongoDB data will NOT persist across restarts."
    VOLUME_PATH="/tmp"
fi
echo "Volume: $VOLUME_PATH"

MONGO_DATA="$VOLUME_PATH/mongodb_data"
mkdir -p "$MONGO_DATA"

# ── 1. Install MongoDB 7 ─────────────────────────────────────
if ! command -v mongod &>/dev/null; then
    echo "Installing MongoDB 7..."
    apt-get update
    apt-get install -y gnupg curl

    # MongoDB 7.0 repo for Ubuntu/Debian
    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
        gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg

    # Detect OS
    . /etc/os-release
    if [[ "$ID" == "ubuntu" ]]; then
        echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu ${VERSION_CODENAME}/mongodb-org/7.0 multiverse" \
            > /etc/apt/sources.list.d/mongodb-org-7.0.list
    else
        echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main" \
            > /etc/apt/sources.list.d/mongodb-org-7.0.list
    fi

    apt-get update
    apt-get install -y mongodb-org
    echo "MongoDB installed: $(mongod --version | head -1)"
else
    echo "MongoDB already installed: $(mongod --version | head -1)"
fi

# ── 2. Start MongoDB ─────────────────────────────────────────
if pgrep -x mongod &>/dev/null; then
    echo "MongoDB already running"
else
    echo "Starting MongoDB (data: $MONGO_DATA)..."
    mongod --dbpath "$MONGO_DATA" --bind_ip_all --port 27017 --fork --logpath /var/log/mongod.log

    # Wait for ready
    for i in $(seq 1 20); do
        if mongosh --quiet --eval "db.runCommand({ping:1})" &>/dev/null; then
            echo "MongoDB is ready!"
            break
        fi
        [ "$i" -eq 20 ] && { echo "ERROR: MongoDB did not start"; exit 1; }
        sleep 1
    done

    # Create admin user (first time only)
    mongosh --quiet --eval "
        try {
            db.getSiblingDB('admin').createUser({
                user: 'admin',
                pwd: '${MONGO_ROOT_PASSWORD:-maripf1admin}',
                roles: ['root']
            });
            print('Admin user created');
        } catch(e) {
            if (e.code === 51003) print('Admin user already exists');
            else print('User creation: ' + e.message);
        }
    "
fi

# ── 3. Install & start mongo-express ─────────────────────────
if ! command -v npx &>/dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

if ! npm list -g mongo-express &>/dev/null 2>&1; then
    echo "Installing mongo-express..."
    npm install -g mongo-express
fi

# Kill existing mongo-express if running
pkill -f "mongo-express" 2>/dev/null || true

echo "Starting mongo-express on port 8081..."
ME_CONFIG_MONGODB_URL="mongodb://localhost:27017/" \
ME_CONFIG_BASICAUTH="false" \
ME_CONFIG_SITE_BASEURL="/" \
    npx mongo-express &>/var/log/mongo-express.log &
echo "  PID: $!"

# ── 4. Run Atlas migration (if not already done) ─────────────
MARKER="$MONGO_DATA/.migration_complete"
if [ ! -f "$MARKER" ]; then
    echo ""
    echo "Running data migration from Atlas..."

    # Install pymongo if needed
    pip install pymongo python-dotenv 2>/dev/null || true

    LOCAL_URI="mongodb://localhost:27017/marip_f1" \
        python3 scripts/migrate_atlas_to_local.py

    touch "$MARKER"
    echo "Migration marker set. Won't re-run on next start."
else
    echo "Migration already completed (marker found). Skipping."
fi

# ── 5. Start the app ─────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo "  MongoDB:       localhost:27017 (data: $MONGO_DATA)"
echo "  Mongo Express: http://localhost:8081"
echo ""
echo "To start the backend:"
echo "  cd $(dirname "$0")/.."
echo "  MONGODB_URI=mongodb://localhost:27017/marip_f1 python pipeline/chat_server.py"
echo ""
echo "To start the frontend:"
echo "  cd frontend && npm run dev"
