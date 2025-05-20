from bliss import setup_globals
from bliss.setup_globals import *
import gevent
import time
import datetime
import sys
import numpy as np
from xmlrpc.client import ServerProxy
from gevent import sleep


from id10utils.bliss import get_from_bliss

def now():
    return str(datetime.datetime.now())

def myprint(*args):
    print(now(),*args)

def _eiger4m_get_server_status():
    server = ServerProxy('http://lid10eiger2lima:9001/RPC2')
    info = server.supervisor.getProcessInfo("LIMA:Eiger4M_V2")
    return info["statename"]

def eiger4m_fix(restart_server_if_running=False):
    server = ServerProxy('http://lid10eiger2lima:9001/RPC2')
    info = server.supervisor.getProcessInfo("LIMA:Eiger4M_V2")
    if restart_server_if_running and _eiger4m_get_server_status() not in ("STOPPED","EXITED"):
        server.supervisor.stopProcess("LIMA:Eiger4M_V2")
        while _eiger4m_get_server_status() not in ("STOPPED","EXITED"):
            time.sleep(1)
    time.sleep(10)
    info = server.supervisor.getProcessInfo("LIMA:Eiger4M_V2")
    if info["statename"] != "RUNNING":
        server.supervisor.startProcess("LIMA:Eiger4M_V2")
        time.sleep(5)
        while _eiger4m_get_server_status() != "RUNNING":
            time.sleep(1)
    time.sleep(10)
    eiger4m_v2.stop()
    eiger4m_v2._det.sync_hard()



def wait_for_temp(verbose=True):
    # wait until setpint is stable
    nanodac_eh2_sample = setup_globals.config.get("omega_sample")
    gevent.sleep(1)
    setpoint = nanodac_eh2_sample.setpoint
    while np.abs(setpoint-nanodac_eh2_sample.axis_position())>0.1:
        #myprint("%3.3f" % nanodac_eh2_sample.axis_position())
        gevent.sleep(2)
    if verbose: myprint(f"End of ramp to {setpoint}")


def set_nanodac_temp(set_temp,ramprate=None,wait=False):
    nanodac_eh2_sample = setup_globals.config.get("omega_sample")
    nanodac_eh2_body = setup_globals.config.get("omega_body")
    temp0 = nanodac_eh2_sample.axis_position()
    t0 = time.time()
    myprint("Starting temperature %g C" % temp0)
    if ramprate is not None:
        nanodac_eh2_sample.ramprate = ramprate
        nanodac_eh2_body.ramprate = ramprate 
        gevent.sleep(1)
    nanodac_eh2_sample.setpoint = set_temp
    nanodac_eh2_body.setpoint = min(set_temp,380)
    myprint(f"Set temperature {set_temp:.2f} C (ramprate {omega_sample.ramprate:.2f} C/min)") 
    if wait: wait_for_temp()


def get_nanodac_temp():
    nanodac_eh2_sample = setup_globals.config.get("omega_sample")
    temp = nanodac_eh2_sample.input.read()
    return round(temp,3)

def ramp(temp,ramprate=None,hold_time=None):
    set_nanodac_temp(temp,ramprate=ramprate,wait=True)
    if hold_time is not None: hold(hold_time)

def hold(delay_min):
    myprint("Starting holding for",delay_min,"minutes")
    sleep(delay_min*60)    
    myprint("End holding")

def _split_ramp(temp,fast_ramprate=5,slow_ramprate=2,dT=2):
    ramp(temp+dT,ramprate=fast_ramprate)
    ramp(temp,ramprate=slow_ramprate)

def split_ramp_down(temp,fast_ramprate=20,slow_ramprate=2,dT=3):
    return _split_ramp(temp,fast_ramprate=fast_ramprate,slow_ramprate=slow_ramprate,dT=abs(dT))

def split_ramp_up(temp,fast_ramprate=20,slow_ramprate=2,dT=3):
    return _split_ramp(temp,fast_ramprate=fast_ramprate,slow_ramprate=slow_ramprate,dT=-abs(dT))

def restore_untulators():
    u35a = config.get("u35a")
    u27b = config.get("u27b")
    u27c = config.get("u27c")
    for i in range(3):
        umv(u35a,13.04)
        umv(u27b,16.13)
        umv(u27c,15.90)

def switch_to_diode():
    m = get_from_bliss("m")
    m.det2.restore_positions("diode",confirm=False)
    select_diode()
    beamstop_out()

def switch_to_eiger():
    ACTIVE_MG.disable("tetramm_diodes*")
    umv(delcoup,2)
    disdiode_saxs()
    beamstop_in()
    eh2_att(1)

def switch_to_transmission():
    ACTIVE_MG.enable("tetramm_diodes*")
    umv(delcoup,0)
    switch_to_diode()
    endiode_saxs()
    eh2_att(0.01)

def select_diode():
    plotselect("tetramm_diodes:ch2_saxs")

def beamstop_in():
    umv(bst,1.5)

def beamstop_out():
    eh2_att(1e-8)
    umv(bst,-2)


def take_data_and_move(nimages,dt=0.01,dz=0.02,n_moves=4):
    period = dt+120e-6
    initial_zs = zs.position
    def wait_and_move():
        for _i in range(n_moves-1):
            gevent.sleep(period*nimages/n_moves)
            umvr(zs,dz)
            myprint(f"New zs position {zs.position:.3f}")
    gwait_and_move = gevent.spawn(wait_and_move)
    mtimescan(dt,nimages,1)
    umv(zs,initial_zs)
    print("Waiting for backgroud job to finish ...",end="")
    gwait_and_move.join()
    print("done")
    return gwait_and_move

def take_data_while_ramping(T,dt=0.01,ramprate=1,exp_time_pilatus=1,hold_time_mins=0):
    currentT = round(get_nanodac_temp(),0)
    dT = T-currentT
    period = dt+100e-6
    secs_in_one_min = 60
    data_collection_time = dT/ramprate+hold_time_mins
    nimages = int(data_collection_time*secs_in_one_min/period)
    nimages_pilatus = data_collection_time/exp_time_pilatus
    if nimages_pilatus > 60_000:
        print("Cannot continue, you have asked for too many images for the pilatus, please increase its exp_time")
        return
    ans=input("Ready to start ? enter 'y' to start ")
    if ans.lower().strip() == "y":
        user.set_nanodac_temp(T,ramprate=ramprate,wait=False)
        mtimescan(dt,nimages,exp_time_pilatus)	
        
def GeO2_6_macro():
    temperatures = 730, 660, 590
    rate = 5,3,3,3,3,3,3
    i = 1
    for temperature in temperatures:
        print(f"set temperature = {temperature}C")
        set_nanodac_temp(temperature,ramprate=3,wait=True)
        time.sleep(20*60)
        print(f"reached temperature = {temperature}C")
        newsample(f"GeO2_6_{temperature}C")
        if True:
            print("check position")
            switch_to_transmission()
            eh2_att(0.01)
            dscan(ys,-0.5,0.5,50,0.2)
            eh2_att(0.001)
            eh2_att(0.01)
            dscan(zs,-0.25,0.25,50,0.2)
            goto_cen()
            eh2_att(0.001)
            switch_to_eiger()  
        mtimescan(0.002,200_000,1)
        i = i + 1
 
def GeO2_6_qdep():
    a_delcoups =  5,3
    theta_s =  2.4,1.5
    i = 1
    for a_delcoup in a_delcoups:
        newsample(f"GeO2_6q_delcoup_{a_delcoup}")
        print(f"go to delcoup = {a_delcoup} and sample = {theta_s[i-1]}")
        umv(delcoup,a_delcoup)
        umv(th,theta_s[i-1])
        sct()
        mtimescan(0.001,3_600_000,1)
        i = i + 1

def GeO2_7_macro():
    temperature  = 660, 520, 485, 450, 415, 380, 300,  30,  30
    rate         =  10,   5,   5,   5,   5,   5,   5,   5,   5
    measure_time =  60, 120,  60,  60,  60,  60,  30,  30, 180

     
    for ii in range(len(temperature)):
        print(f"Set temperature = {temperature[ii]}C")
        set_nanodac_temp(temperature[ii], ramprate=rate[ii], wait=True)
        
        time.sleep(5*60)
        
        print(f"Reached temperature = {temperature[ii]}C")
        newsample(f"GeO2_7_{temperature[ii]}C")
        print(f"new measure")
        switch_to_transmission()
        eh2_att(0.01)
        dscan(ys, -0.5, 0.5, 50, 0.2)
        eh2_att(0.001)
        eh2_att(0.01)
        dscan(zs, -0.5, 0.5, 50, 0.2)
        eh2_att(0.001)
        eh2_att(0.01)
        dmesh(ys, -0.05, 0.05, 20, zs, -0.05, 0.05, 20, 0.2)
        eh2_att(0.001)
        switch_to_eiger()

        if ii==len(temperature)-1:
            eh2_att(0.5)
       
        mtimescan(0.001, measure_time[ii]*60*1000, 1)
        switch_to_transmission()
        eh2_att(0.01)
        dscan(ys, -0.5, 0.5, 200, 0.2)
        eh2_att(0.001)
        eh2_att(0.01)
        dscan(zs, -0.5, 0.5, 200, 0.2)
        eh2_att(0.001)
        eh2_att(0.01)
        dmesh(ys, -0.05, 0.05, 20, zs, -0.05, 0.05, 20, 0.2)
        eh2_att(0.001)
     
def GeO2_7_discesa():
    temperature = 520, 300,  30
    rate = 100, 100, 100
    time =  20,  10,  10

    for ii in range(len(temperature)):
        set_nanodac_temp(temperature[ii], ramprate=rate[ii], wait=True)
        print(f"Reached temperature = {temperature[ii]}C")
        newsample(f"GeO2_7_{temperature[ii]}C")
        print(f"new measure")
        switch_to_transmission()
        eh2_att(0.01)
        dscan(ys, -0.5, 0.5, 50, 0.2)
        eh2_att(0.001)
        eh2_att(0.01)
        dscan(zs, -0.5, 0.5, 50, 0.2)
        eh2_att(0.001)
        eh2_att(0.01)
        dmesh(ys, -0.05, 0.05, 20, zs, -0.05, 0.05, 20, 0.2)
        eh2_att(0.001)
        switch_to_eiger()

        mtimescan(0.001, time[ii]*60*1000, 1)


def GeO2_3_macro():
    temperature  =  30, 100, 170, 240, 310, 380, 415, 450, 485, 520, 555, 590, 625, 660, 695, 730, 730, 660, 590, 520, 450, 380, 310, 240, 170, 100, 30, 30
    rate         =  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10, 10, 10
    measure_time =  25,  25,  25,  25,  25,  50,  50,  50,  50, 100, 100, 100, 150, 150, 150, 180,  180, 20,  20,  20,  10,  10,  10,  10,  10,  10, 10, 20
     
    umv(ys, 0.1, zs, 0)
    for ii in range(len(temperature)):
        if (ii!=16) or (ii!=len(temperature)-1):
            print(f"Set temperature = {temperature[ii]}C")
            set_nanodac_temp(temperature[ii], ramprate=rate[ii], wait=True)
            print(f"Reached temperature = {temperature[ii]}C")

            newsample(f"GeO2_3_{temperature[ii]}C")
        print(f"new measure")

        switch_to_transmission()
        dscan(ys, -0.5, 0.5, 50, 0.2)
        dscan(zs, -0.5, 0.5, 50, 0.2)
        dmesh(ys, -0.05, 0.05, 20, zs, -0.05, 0.05, 20, 0.2)

        eh2_att(0.001)

        if ii == 16:
            eh2_att(0.1)
        if ii == len(temperature)-1:
            eh2_att(0.5)

        switch_to_eiger()

        mtimescan(0.001, measure_time[ii]*60*1000, 1)


def GeO2_3_macro2():
    temperature  =  660, 695, 730
    rate         =   10,  10,  10
    measure_time =   40,  40,  30
    num_rep      =    4,   4,   5
    yss          =   .0, .05, .1,  .15,  .2          
    zss          =   .1, .05, .0, -.05, -.01
    
    #umv(ys, 0.1, zs, 0) working point
    for ii in range(3):

        print(f"Set temperature = {temperature[ii]}C")
        set_nanodac_temp(temperature[ii], ramprate=rate[ii], wait=True)
        print(f"Reached temperature = {temperature[ii]}C")

        newsample(f"GeO2_3_{temperature[ii]}C")
        print(f"new measure")

        umv(ys, yss[ii], zs, zss[2])
        
        switch_to_transmission()
        dscan(ys, -0.5, 0.5, 100, 0.2)
        dscan(zs, -0.5, 0.5, 100, 0.2)
        dmesh(ys, -0.05, 0.05, 10, zs, -0.15, 0.15, 30, 0.2)

        eh2_att(0.001)

        switch_to_eiger()

        for jj in range(num_rep[ii]):    
            umv(zs, zss[jj])
            mtimescan(0.001, measure_time[ii]*60*1000, 1)
    
 
    delcoups      = 1,   3,  4,   5
    ths           = 2, 1.5,  2, 2.4  
    measure_time = 30,  30, 20,  20

    print("Q measure")
    newsample(f"GeO2_3_730C_Q")
    
    umv(ys, yss[3], zs, zss[2])
    
    switch_to_transmission()
    dscan(ys, -0.5, 0.5, 100, 0.2)
    dscan(zs, -0.5, 0.5, 100, 0.2)
    dmesh(ys, -0.05, 0.05, 10, zs, -0.15, 0.15, 30, 0.2)

    eh2_att(0.001)

    switch_to_eiger()

    for ii in range(len(delcoups)):
        umv(zs, zss[ii])
        umv(delcoup,delcoups[ii])
        umv(th,ths[ii])

        mtimescan(0.001, measure_time[ii]*60*1000, 1)


    print("Attenuator 0.5 measure @ T = 730 C")
    newsample(f"GeO2_3_730C_att2")

    umv(ys, yss[4], zs, zss[2])
    
    switch_to_transmission()
    dscan(ys, -0.5, 0.5, 100, 0.2)
    dscan(zs, -0.5, 0.5, 100, 0.2)
    dmesh(ys, -0.05, 0.05, 10, zs, -0.15, 0.15, 30, 0.2)

    eh2_att(0.001)

    switch_to_eiger()

    eh2_att(0.5)

    for ii in range(4):
        umv(zs, zss[ii])

        mtimescan(0.001, 30*60*1000, 1)
    

    print("Set temperature = 30 C")
    set_nanodac_temp(30, ramprate=30, wait=True)
    time.sleep(10*60)
    print("Reached temperature = 30 C")

    newsample("GeO2_3_30C")
    print("new measure")

    umv(ys, yss[3], zs, zss[4])
    
    switch_to_transmission()
    dscan(ys, -0.5, 0.5, 100, 0.2)
    dscan(zs, -0.5, 0.5, 100, 0.2)
    dmesh(ys, -0.05, 0.05, 20, zs, -0.05, 0.05, 20, 0.2)

    eh2_att(0.001)

    switch_to_eiger()

    mtimescan(0.001, 20*60*1000,1)

    print("Attenuator 0.5 measure @ T = 30 C")

    eh2_att(0.5)
    mtimescan(0.001, 40*60*1000,1)
    

def GeO2_4_macro():
    print("Set temperature = 30 C")
    set_nanodac_temp(30, ramprate=30, wait=True)
    
    print("Reached temperature = 30 C")

    newsample("GeO2_4_30C")
    print("new measure")
    
    dscan(ys, -0.5, 0.5, 100, 0.2)
    dscan(zs, -0.5, 0.5, 100, 0.2)
    dmesh(ys, -0.05, 0.05, 20, zs, -0.05, 0.05, 20, 0.2)

    eh2_att(0.001)

    switch_to_eiger()

    mtimescan(0.001, 20*60*1000,1)

    print("Attenuator 0.5 measure @ T = 30 C")

    eh2_att(0.5)
    mtimescan(0.001, 40*60*1000,1)




