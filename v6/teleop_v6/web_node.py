"""Use the shared WebXR transport with the V6 operator page."""
from pathlib import Path
from openarmx_teleop_vr_306_v5.vr_web_bridge import VrWebBridge


class WebBridge(VrWebBridge):
    async def _serve(self):
        self._web_dir = Path(__file__).resolve().parent / "web"
        await super()._serve()
