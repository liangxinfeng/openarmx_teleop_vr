"""Selection-only squat mode guard; this module never produces joint targets."""
import math


class ManualSquatGuard:
    def __init__(self):
        self.standing_since = None
        self.last_stamp = None

    def observe(self, legs, stamp, now, timeout, selected_reverse=False, busy=False):
        result = {'supported': True, 'mode': None, 'can_switch': False,
                  'message': '等待新鲜腰腿反馈', 'feedback_time_monotonic': stamp}
        try:
            ankle, knee = float(legs[0]), float(legs[1])
            stamp, now, timeout = float(stamp), float(now), float(timeout)
            if not all(math.isfinite(v) for v in (ankle, knee, stamp, now, timeout)):
                raise ValueError('non-finite feedback')
            if not 0 <= now-stamp <= min(.5, timeout):
                raise ValueError('stale feedback')
        except (TypeError, ValueError, IndexError):
            self.standing_since = self.last_stamp = None
            return result
        lowering = .56*(ankle*76.+knee*155.)/(76.**2+155.**2)
        reverse = bool(selected_reverse) if abs(lowering) <= .002 else lowering < 0
        result['mode'] = 'reverse' if reverse else 'normal'
        standing = max(abs(ankle), abs(knee)) <= 1.0 and abs(lowering) <= .002
        if not standing or busy:
            self.standing_since = None
        elif (self.last_stamp is None or stamp < self.last_stamp
              or stamp-self.last_stamp > min(.5, timeout)):
            self.standing_since = stamp
        elif self.standing_since is None:
            self.standing_since = stamp
        self.last_stamp = stamp
        if busy:
            result['message'] = '请先停止升降，并等待复位或姿态调整结束'
        elif not standing:
            result['message'] = '请先按 Y 升回直立，再切换下蹲方式'
        elif stamp-self.standing_since < .35:
            result['message'] = '等待腰腿直立停稳'
        else:
            result.update(can_switch=True, message='可切换；选择模式不会触发运动')
        return result

    @staticmethod
    def validate_request(payload, now):
        mode = payload.get('select_squat_mode')
        request_id = payload.get('request_id')
        deadline = payload.get('expires_monotonic')
        if mode not in ('normal', 'reverse'):
            raise ValueError('下蹲方式必须为正向或反向')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 96:
            raise ValueError('下蹲方式请求标识无效')
        if payload.get('enabled') is not False or payload.get('direction') != 0:
            raise ValueError('切换方式不能同时携带升降指令')
        if any(k in payload for k in ('target_lowering_m', 'target_level')):
            raise ValueError('切换方式不能同时设置身体高度')
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise ValueError('下蹲方式请求已失效')
        if not math.isfinite(deadline) or not 0 <= deadline-now <= 2.0:
            raise ValueError('下蹲方式请求已过期，请重新选择')
        return mode, request_id
