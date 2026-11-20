#!/usr/bin/env bash
set -eo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
migration="${V6_PLATFORM_ROOT:-$HOME/migration367_20261118}"
source "$migration/releases/control-center-current/environment.sh"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
export V6_TOPIC_SUFFIX="${V6_TOPIC_SUFFIX:-0_367}"
if [[ "${1:-}" == "--simulate" ]]; then
  export V6_TEST_OUTPUT=$(mktemp -d "${TMPDIR:-/tmp}/teleop-v6-test.XXXXXXXX")
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
  cd "$V6_TEST_OUTPUT"
  echo "Simulation reports: $V6_TEST_OUTPUT"
  python -m pytest -q -p no:cacheprovider "$here/test_boundaries.py"
  python "$here/test_tracking.py" >tracking.log
  python "$here/test_arm_shape.py" >arm_shape.log
  python "$here/test_forearm_pose.py" --verify >forearm_pose.jsonl
  python "$here/test_straight_return.py" --verify >straight_return.jsonl
  python "$here/test_return.py" >return_replay.log
  python "$here/test_regrip_return.py" --verify >return_reference.jsonl
  bash "$here/loop.sh"
  bash "$here/run_ros_regrip_replay.sh"
  exit
fi
[[ $# == 0 ]] || { echo 'usage: run_v6.sh [--simulate]' >&2; exit 2; }
# Match the installed V5 local motor bridge discovery profile.
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
python "$here/build.py" --check
exec python "$here/desktop_session.py"
