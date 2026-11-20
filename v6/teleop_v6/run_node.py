"""Entrypoint for V6 nodes, including the compatible WebXR UI."""
import sys
import signal


def main():
    role = sys.argv.pop(1)
    import rclpy
    if role == "controller":
        from .controller import Controller as Node
    elif role == "mapper":
        from .compat.vr_mapper_node import IndependentVrMapper as Node
    elif role == "web":
        from .web_node import WebBridge as Node
    else:
        raise ValueError(role)
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    node = Node()
    try:
        while not stopping and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.05)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if role == "controller" and rclpy.ok():
                node.finish()
        finally:
            try:
                node.destroy_node()
            finally:
                if rclpy.ok():
                    rclpy.shutdown()


if __name__ == "__main__":
    main()
