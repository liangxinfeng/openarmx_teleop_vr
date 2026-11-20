"""Drive the production ROS nodes against synthetic vendor endpoints in an isolated domain."""
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import ssl
import signal
import time
import urllib.request
from pathlib import Path

import numpy as np
import aiohttp
import rclpy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from openarmx_teleop_vr_306_v5.schema import GROUP_FIELDS

if os.environ.get("ROS_DOMAIN_ID")!="191" or os.environ.get("V6_LOOPBACK_TEST")!="1":
    raise RuntimeError("Run only in the dedicated simulation domain")
rclpy.init()
node=rclpy.create_node("v6_virtual_motors")
base="/teleop_v6/"
def sub(t,fn):
    return node.create_subscription(String,t,lambda m:fn(m.data),10)
def pub(t):return node.create_publisher(String,t,10)
def hardware(t):return "/"+t+"_v6_sim"

home={k:np.zeros(n) for k,(_,n) in GROUP_FIELDS.items()}
home["left_arm"]=np.array([20.,0.,0.,-110.,0.,0.,0.])
home["right_arm"]=home["left_arm"].copy()
q={k:v.copy() for k,v in home.items()}
for arm in ("left_arm","right_arm"):
    q[arm][3]+=8.0
goal={k:v.copy() for k,v in q.items()}
status={};states=[]; events=[]; synced=False; grip=False; offset=0.
body_packet={}
stage_times=[]
moving_until=0.
armed_while_moving=False
def backend(data):
    global armed_while_moving
    status.update(json.loads(data))
    if moving_until>time.monotonic() and status.get("hardware_enabled"):
        armed_while_moving=True
    pair=(status.get("state"),status.get("hardware_enable_stage"))
    if not states or states[-1]!=pair:
        states.append(pair)
        stage_times.append((pair,time.monotonic()))
def command(data):
    for k,v in json.loads(data).items():
        group=k.removesuffix("_target_joints_position")
        if group in goal:goal[group]=np.asarray(v,float)
def reset(data):
    events.append("full_reset")
    for k,v in home.items():goal[k]=v.copy()
def sync(req,res):
    global synced,moving_until
    synced=bool(req.data)
    events.append("sync_on" if synced else "sync_off")
    if synced and events.count("sync_on")==2:
        moving_until=time.monotonic()+.35
    res.success=True;res.message="virtual vendor acknowledged"
    return res
def packet():
    p=lambda x,y,z:dict(x=x,y=y,z=z)
    quat=dict(x=0.,y=0.,z=0.,w=1.)
    def hand(sign):return dict(position=p(sign*.22,.9,-.35-offset),quaternion=quat,gripActive=grip,trigger=0.)
    return dict(head=dict(position=p(0.,1.6,0.),rotation=p(0.,0.,0.),quaternion=quat),leftController=hand(-1),rightController=hand(1),armLandmarks=body_packet)
fb=pub(hardware("topic_arm_whole_body_and_gripper_current_joints_status"))
sub(base+"status",backend)
sub(hardware("topic_arm_whole_body_target_joints_position"),command)
sub(hardware("control_reset"),reset)
for t in ("topic_arm_gripper_target_joints_position","topic_arm_joints_set_enable_state","topic_arm_joints_clear_error","control_prepare_arms_only","control_independent_sync_heartbeat","control_independent_sync_session_command"):
    sub(hardware(t),lambda d:None)
node.create_service(SetBool,hardware("control_independent_sync_hold_session"),sync)
enabled=node.create_client(SetBool,base+"set_hardware_enabled")
quick=node.create_client(Trigger,base+"quick_reset")
def tick():
    msg={}
    for k,(field,_) in GROUP_FIELDS.items():
        delta=(goal[k]-q[k])*.22
        q[k]+=delta
        msg[field]=dict(position=q[k].tolist(),speed=(delta/.01).tolist())
        if k=="left_arm" and moving_until>time.monotonic():
            msg[field]["speed"][0]=8.
        msg[k+"_target_joint_state"]=goal[k].tolist()
    fb.publish(String(data=json.dumps(msg)))
node.create_timer(.01,tick)
def wait(test,sec,why):
    until=time.monotonic()+sec
    while time.monotonic()<until:
        rclpy.spin_once(node,timeout_sec=.015)
        if test():return
    raise AssertionError((why,events,states[-9:],status.get("reason"),status.get("detail")))
def pause(sec):
    start=time.monotonic();wait(lambda:time.monotonic()-start>=sec,sec+1,"pause")
def call(client,request):
    f=client.call_async(request)
    wait(f.done,5,"service reply")
    r=f.result()
    assert r.success,r.message
    return r.message

stop_stream=threading.Event()
async def headset():
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect("https://127.0.0.1:18448/ws?client_id=v6-simulator",ssl=False) as ws:
            sequence=0
            while not stop_stream.is_set():
                sequence+=1
                await ws.send_json({**packet(),"sequence":sequence})
                await asyncio.sleep(1/90)

def web_enable():
    request=urllib.request.Request("https://127.0.0.1:18448/api/hardware-control",
        data=b'{"enabled":true}',headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request,context=ssl._create_unverified_context(),timeout=30) as response:
        result=json.load(response)
        assert result["success"],result

pool=ThreadPoolExecutor(max_workers=2)

try:
    wait(lambda:enabled.service_is_ready() and status.get("body_height",{}).get("feedback_fresh"),18,"ready")
    with urllib.request.urlopen("https://127.0.0.1:18448/",context=ssl._create_unverified_context(),timeout=3) as resp:
        assert resp.status==200 and "V6" in resp.read().decode()
    stream=pool.submit(asyncio.run,headset())
    pause(1.15)  # The shared command monitor must observe one full quiet second.
    for cycle in range(2):
        if cycle==0:
            start=time.monotonic()
            enable_page=pool.submit(web_enable)
            wait(enable_page.done,30,"web enable")
            enable_page.result()
            timing=dict(first_enable_sec=round(time.monotonic()-start,3), stages=[
                [state,stage,round(stamp-start,3)] for (state,stage),stamp in stage_times if stamp>=start])
            print(json.dumps(timing),flush=True)
            Path("first_enable.json").write_text(json.dumps(timing,indent=2))
            if os.environ.get("V6_BASELINE_TIMING")=="1":
                call(enabled,SetBool.Request(data=False))
                wait(lambda:not synced,5,"baseline release")
                os.kill(int(os.environ["V6_TEST_LAUNCH_PID"]),signal.SIGINT)
                raise SystemExit(0)
        wait(lambda:events.count("sync_on")==cycle+1 and status.get("hardware_enabled") and synced,28,"enabled")
        if cycle:
            pause(.25)
        grip_start=time.monotonic()
        grip=True
        wait(lambda:status.get("ik_session_v5",{}).get("active")==["left","right"],3,"first accepted Grip")
        print(json.dumps(dict(cycle=cycle+1,first_grip_observed_ms=round(1000*(time.monotonic()-grip_start)))),flush=True)
        offset=.035;pause(.9)
        movement=[float(np.max(abs(q[s]-home[s]))) for s in ("left_arm","right_arm")]
        assert min(movement)>.8,(movement,status.get("reason"))
        offset=0.;pause(1.2)
        returned=[float(np.max(abs(q[s]-home[s]))) for s in ("left_arm","right_arm")]
        print("manual return joint residual deg",returned,flush=True)
        assert max(returned)<1.1,returned
        if cycle==0:
            def shape_source(value):
                arms=status.get("ik_session_v5",{}).get("retargeting",{})
                return all(arms.get(s,{}).get("shape_source")==value for s in ("left","right"))
            body_packet={s:[[sign*.2,1.3,0],[sign*.24,1.08,-.08],[sign*.2,1.10,-.35]]
                         for s,sign in (("left",-1),("right",1))}
            wait(lambda:shape_source("body"),3,"both arm observations reached IK")
            body_packet={}
            wait(lambda:all(status.get("ik_session_v5",{}).get("retargeting",{}).get(s,{}).get("shape_source")
                         in ("neutral","raised_wrist") for s in ("left","right")),3,"recovered after body tracking loss")
            pause(.9)
        offset=.055;pause(.9)
        grip=False;pause(.35)
        call(quick,Trigger.Request())
        wait(lambda:status.get("quick_reset_active"),3,"quick reset started")
        wait(lambda:not status.get("quick_reset_active"),25,"quick reset finished")
        assert max(np.max(abs(q[k]-home[k])) for k in ("left_arm","right_arm"))<1.1
        offset=0.
        call(enabled,SetBool.Request(data=False))
        if cycle==0:
            print("rapid reentry",call(enabled,SetBool.Request(data=True)),flush=True)
        else:
            wait(lambda:not status.get("hardware_enabled") and not synced,6,"exit ack")
        print("completed",cycle+1,movement,flush=True)
    assert events.count("full_reset")==2 and events.count("sync_on")==2,events
    assert not armed_while_moving,"second entry accepted motion after SYNC"
    call(enabled,SetBool.Request(data=True))
    wait(lambda:events.count("full_reset")==3,8,"third reset issued")
    call(enabled,SetBool.Request(data=False))
    wait(lambda:status.get("state")=="EXITING",2,"reset draining")
    pause(.45)
    call(enabled,SetBool.Request(data=True))
    wait(lambda:events.count("sync_on")==3 and status.get("hardware_enabled"),15,"resume existing reset")
    assert events.count("full_reset")==3
    os.kill(int(os.environ["V6_TEST_LAUNCH_PID"]),signal.SIGINT)
    wait(lambda:not synced,7,"process close releases active session")
    print(json.dumps(dict(tracking_cycles=2,enabled_sessions=3,events=events,states=states,closed=not synced)),flush=True)
finally:
    stop_stream.set()
    pool.shutdown(wait=True)
    node.destroy_node();rclpy.shutdown()
