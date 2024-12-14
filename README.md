# go1-rl-kit
![Image1](/cover.png)
Deployment kit for Unitree Go1 Edu 

This repo can be deployed in the following configurations:
1. External PC/NUC with LAN (Prefered) (Tested)
2. External PC/NUC with WiFi 
3. Internal Computer of the Go1 

# Dependencies 
- python3.8
- torch
- matplotlib
- numpy<1.24
- scipy
- other dependencies for unitree_legged_sdk

# Installation 
On the target machine run the following commands
1. `git clone https://github.com/TextZip/go1-rl-kit`  
2. `cd go1-rl-kit`
3. Build the unitree_legged_sdk using instructions from https://github.com/unitreerobotics/unitree_legged_sdk/tree/v3.8.0
4. Make sure to run one of the python examples from the `unitree_legged_sdk/example_py/` to make sure the build is working. 
5. Manually assign the following network config
    - IP Address: 192.168.123.162
    - Subnet Mask: 255.255.255.0
    - Default Gateway: 192.168.123.1
6. `ping 192.168.123.10` to make sure you are able to reach the motor controller

# Deployment
1. Turn on the robot and the unitree joystick and wait for it to automatically stand up.
2. Enter low level mode using the following commands:
    - L2 + A
    - L2 + A
    - L2 + B
    - L1 + L2 + Start
3. Connect the LAN cable to the robot and make sure you can ping the motor controller `ping 192.168.123.10` 
4. Suspend the robot in the air 
5. Run `python deploy.py` while the robot is suspended in the air
2. While suspended, please wait for the robot to reach a standing pose (2-3 Seconds) and then slowly put the robot back on the ground
3. Do not operate the joystick or the robot until the counter on the screen reaches 2300, after the counter reaches 2300 you can move the robot around. 
4. Left joystick can be used for giving linear velocity commands, the right joystick can be used to give yaw commands.


