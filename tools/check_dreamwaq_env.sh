#!/usr/bin/env bash
set -euo pipefail

CONTAINER="dreamwaq-run"
DO_FIX=0

usage() {
  cat <<'EOF'
Usage:
  ./tools/check_dreamwaq_env.sh [--container NAME] [--fix]

Options:
  --container NAME   Docker container name (default: dreamwaq-run)
  --fix              Reinstall editable packages inside container:
                     /home/user/dreamwaq/rsl_rl and /home/user/dreamwaq/legged_gym
  -h, --help         Show this help

Exit codes:
  0  Environment is correct
  1  Usage / runtime error
  2  Environment check failed (wrong import path or RMS flags)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container)
      CONTAINER="$2"
      shift 2
      ;;
    --fix)
      DO_FIX=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found" >&2
  exit 1
fi

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "Container not found: $CONTAINER" >&2
  exit 1
fi

RUNNING="$(docker inspect -f '{{.State.Running}}' "$CONTAINER")"
if [[ "$RUNNING" != "true" ]]; then
  echo "Starting container: $CONTAINER"
  docker start "$CONTAINER" >/dev/null
fi

if [[ "$DO_FIX" -eq 1 ]]; then
  echo "Reinstalling editable packages inside container..."
  docker exec "$CONTAINER" bash -lc '
    set -euo pipefail
    python -m pip uninstall -y legged-gym legged_gym rsl-rl rsl_rl >/dev/null 2>&1 || true
    python -m pip install -e /home/user/dreamwaq/rsl_rl
    python -m pip install -e /home/user/dreamwaq/legged_gym
  '
fi

echo "Checking runtime package path and go2_waq RMS flags..."
docker exec "$CONTAINER" bash -lc 'python - <<"PY"
import sys
import legged_gym
from legged_gym.envs.go2.go2_config import Go2RoughCfgWaqPPO

pkg_path = legged_gym.__file__
obs = Go2RoughCfgWaqPPO.runner.obs_rms
priv = Go2RoughCfgWaqPPO.runner.privileged_obs_rms
vel = Go2RoughCfgWaqPPO.runner.true_vel_rms

ok_path = pkg_path.startswith("/home/user/dreamwaq/legged_gym/")
ok_flags = (obs is False) and (priv is False) and (vel is False)

print("legged_gym path:", pkg_path)
print("obs_rms:", obs)
print("privileged_obs_rms:", priv)
print("true_vel_rms:", vel)

if not ok_path:
    print("\nFAIL: Python is not importing legged_gym from /home/user/dreamwaq/legged_gym", file=sys.stderr)
if not ok_flags:
    print("FAIL: go2_waq RMS flags are not all False", file=sys.stderr)

if not (ok_path and ok_flags):
    sys.exit(2)

print("\nPASS: Environment is aligned for no-RMS go2_waq training")
PY'

echo
echo "Recommended train command:"
echo "docker exec -it $CONTAINER bash -lc 'cd /home/user/dreamwaq/legged_gym/legged_gym/scripts && python train.py --task=go2_waq --headless'"
