"""原厂扩展状态机回归。所有关节、ROS、时钟均为内存替身。"""
import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
import threading
import json

class ControlState(Enum):
    ASYNC=0; HOME=1; SYNC=2

class Clock:
    def __init__(self): self.t=1.
    def monotonic(self): return self.t
    def sleep(self,s): self.t+=s
    def now(self): return self.t

class Joint:
    def __init__(self,n): self.q=[12.]*n;self.target=[13.]*n;self.mode=None
    def get_current_joint_positions(self): return self.q[:]
    def get_current_joint_velocities(self): return [0.]*len(self.q)
    def get_target_joint_positions(self): return self.target[:]
    def set_target_joint_positions(self,p): self.target=list(p)
    def set_control_mode(self,m): self.mode=m

class Manager:
    def __init__(self):
        for key,n in [('neck_joint_state',3),('leg_waist_joint_state',4),('left_arm_joint_state',7),('right_arm_joint_state',7),('left_gripper_state',1),('right_gripper_state',1)]:
            setattr(self,key,Joint(n))
        self.state=ControlState.ASYNC;self.protected=False;self.pid=False
    def get_control_state(self): return self.state
    def set_control_state(self,s): self.state=s
    def is_protected(self): return self.protected
    def set_arm_joints_pid_loop_state(self,v): self.pid=v

class Node:
    def __init__(self,*args): self.services={};self.subscriptions={};self.timers=[]
    def create_service(self,kind,name,cb,**kw): self.services[name]=cb
    def create_subscription(self,kind,name,cb,qos,*a,**kw): self.subscriptions[name]=cb;return cb
    def create_timer(self,period,cb,**kw): self.timers.append(cb)

def setup():
    clock=Clock();published=[]
    def handler(*a):
        return NS(keep_disabling_joints=False,arm_task_control=NS(),previous_timestamp=0,
                  last_received_vr_command_time=0,realtime_clock=clock,
                  pub_arm_joints_clear_error=NS(publish=lambda m:published.append(('clear',m.data))),
                  pub_arm_joints_set_enable_state=NS(publish=lambda m:published.append(('enable',m.data))))
    tree=ast.parse(Path(__file__).with_name('migration_sync_bridge.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    ns=dict(threading=threading,time=clock,json=json,Node=Node,MutuallyExclusiveCallbackGroup=lambda:None,
            ArmVrCmdHandler=handler,PROGRAM_SETTINGS={},ControlState=ControlState,
            ControlMode=NS(POSITION='position'),String=lambda **kw:NS(**kw),SetBool=object)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'bridge','exec'),ns)
    state={};manager=Manager();ns['install'](state,manager,{},'0_367')
    def call(name,active):
        return state['node'].services['/control_independent_sync_'+name+'_0_367'](NS(data=active),NS(success=False,message=''))
    return state,manager,published,clock,call

def test_registration_does_not_enable_or_write_targets():
    state,m,p,clock,call=setup()
    assert not p and m.state==ControlState.ASYNC
    assert m.left_arm_joint_state.target==[13.]*7
    assert len(state['node'].services)==4

def test_hold_entry_and_release():
    state,m,p,clock,call=setup()
    assert call('hold_session',True).success
    assert m.state==ControlState.SYNC and [x[0] for x in p]==['enable']
    assert m.left_arm_joint_state.target==m.left_arm_joint_state.q
    assert call('hold_session',False).success and m.state==ControlState.ASYNC

def test_grasp_finish_keeps_closing_target():
    state,m,p,clock,call=setup();assert call('hold_session',True).success
    m.right_gripper_state.target=[350.];m.right_gripper_state.q=[180.]
    assert call('grasp_finish',False).success
    assert m.right_gripper_state.target==[350.]

def test_loaded_handoff_keeps_gripper_command_on_entry_and_exit():
    state,m,p,clock,call=setup();m.right_gripper_state.target=[300.]
    assert call('loaded_session',True).success
    assert m.right_gripper_state.target==[300.]
    m.right_gripper_state.target=[10.]
    assert call('loaded_session',False).success
    assert m.right_gripper_state.target==[10.]

def test_protection_never_enables():
    state,m,p,clock,call=setup();m.protected=True
    assert not call('hold_session',True).success and not p

def test_invalid_loaded_target_never_enables():
    state,m,p,clock,call=setup();m.right_gripper_state.target=[float('nan')]
    assert not call('loaded_session',True).success and not p

def test_session_mode_conflict_rejected():
    state,m,p,clock,call=setup();assert call('hold_session',True).success
    assert not call('loaded_session',True).success

def test_duplicate_entry_does_not_reenable():
    state,m,p,clock,call=setup();assert call('hold_session',True).success
    assert call('hold_session',True).success and len(p)==1

def test_watchdog_holds_and_exits():
    state,m,p,clock,call=setup();assert call('hold_session',True).success
    clock.sleep(2);state['node'].timers[0]()
    assert m.state==ControlState.ASYNC

def test_official_vr_gate_only_during_owned_session():
    state,m,p,clock,call=setup();events=[]
    cb=state['node'].create_subscription(object,'/control_topic_0_367',events.append,10)
    cb(1);assert events==[1]
    assert call('hold_session',True).success;cb(2);assert events==[1]
    assert call('hold_session',False).success;cb(3);assert events==[1,3]
