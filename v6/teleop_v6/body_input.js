// Same XR frame and local-floor reference space as the controllers.
function readArmLandmarks(xr) {
  const frame = xr.getFrame?.();
  const space = xr.getReferenceSpace?.();
  const arms = {};
  if (!frame || !space) return arms;
  try {
    const body = frame.body;
    if (!body) return arms;
    for (const side of ['left', 'right']) {
      const points = ['arm-upper', 'arm-lower', 'hand-wrist'].map(part => {
        const joint = body.get(side + '-' + part);
        const p = joint && frame.getPose(joint, space)?.transform.position;
        return p && [p.x, p.y, p.z];
      });
      if (points.every(Boolean)) arms[side] = points;
    }
  } catch (error) {
    if (error.name !== 'InvalidStateError' && error.name !== 'NotSupportedError' && error.name !== 'SecurityError') throw error;
  }
  return arms;
}
