import pygame
import threading as threading
import time as t

# Initializing pygame
pygame.init()

# Initializing Joysticks
joystick_0 = pygame.joystick.Joystick(0)
joystick_0.init()

joystick_1 = pygame.joystick.Joystick(1)
joystick_1.init()
deadzone = 0.05
joy_0_xpos = 0.0
joy_0_ypos = 0.0
joy_1_xpos = 0.0
joy_1_ypos = 0.0
trig_0 = 0
trig_1 = 0

# Deadzone constant and values


class ControllerManager:
    def __init__(self):
        self.input_thread = threading.Thread(target=self.input_loop)
        self._input_last_time_step = [
            (joy_0_xpos, joy_0_ypos),
            (joy_1_xpos, joy_1_ypos),
            trig_0,
            trig_1,
        ]
        self.input_thread.start()

    def get_input(self):
        return self._input_last_time_step

    def input_loop(self):
        global joystick_0, joystick_1, deadzone, joy_0_xpos, joy_0_ypos, joy_1_xpos, joy_1_ypos, trig_0, trig_1
        while self.input_thread.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    if event.joy == 0:
                        if event.axis == 0:
                            if abs(event.value) < deadzone:
                                joy_0_xpos = 0.0
                            elif event.value < 0:
                                joy_0_xpos = -1 * (event.value + deadzone)
                            else:
                                joy_0_xpos = event.value - deadzone
                        elif event.axis == 1:
                            if abs(event.value) < deadzone:
                                joy_0_ypos = 0.0
                            elif event.value < 0:
                                joy_0_ypos = -1 * (event.value + deadzone)
                            else:
                                joy_0_ypos = event.value - deadzone
                    else:
                        if event.axis == 0:
                            if abs(event.value) < deadzone:
                                joy_1_xpos = 0.0
                            elif event.value < 0:
                                joy_1_xpos = -1 * (event.value + deadzone)
                            else:
                                joy_1_xpos = event.value - deadzone
                        # Vertical axis
                        elif event.axis == 1:
                            if abs(event.value) < deadzone:
                                joy_1_ypos = 0.0
                            elif event.value < 0:
                                joy_1_ypos = -1 * (event.value + deadzone)
                            else:
                                joy_1_ypos = event.value - deadzone

                # Checking for trigger events
                elif event.type == pygame.JOYBUTTONDOWN:
                    # Joystick 0
                    if event.joy == 0:
                        trig_0 = 1
                    # Joystick 1
                    else:
                        trig_1 = 1

                # Checking for trigger releases
                elif event.type == pygame.JOYBUTTONUP:
                    # Joystick 0
                    if event.joy == 0:
                        trig_0 = 0
                    # Joystick 1
                    else:
                        trig_1 = 0

            # Saving all joystick inputs
            self._input_last_time_step = [
                (joy_0_xpos, joy_0_ypos),
                (joy_1_xpos, joy_1_ypos),
                trig_0,
                trig_1,
            ]

            t.sleep(0.0166)


# Main loop
