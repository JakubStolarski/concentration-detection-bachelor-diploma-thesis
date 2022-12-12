# Concentration Detecion Application (WIP)
This application was made as a thesis for my bachelor diploma. It is a simple application, which uses mediapipe library and basic formulas of machine vision, to allow for a user to monitor their concentration. 

There are three modes at the moment: Configuration, Run and Showcase.

Cofiguration - Allows user to configure their workspace 
Run - Is a basic mode that analyzes concentration based on user set parameters
Showcase - Shows POI taken by the program, approximated viewing direction and time that passes

# Configuration method
1. Run flaskend.py
2. Select Cofiguration mode, set it parameters and submit
3. Put your thumb up (if it is detected 'Okay!' will be written on video feed)
4. Create a rectangle "with your face" - stare at one of the corners of yor workspace and then move onto another by moving your face towards it
5. Change the mode to Run and submit again
6. If you wish you can click a button Save&Quit That will save your current workspace for later use

# Parameters
1. Cam ID and Monitor ID are self explanatory
2. Distraction time is the time it takes the script to take action, e.g if it is set to 30, it will take 30 seconds before staring out of the workspace, will be considered lack of concentration and warning will pop-up
3. For Detection and Tracing Confidence please refer to Mediapipe documentation: https://google.github.io/mediapipe/

# Bibliography
The whole bibliography can be found in library.bib file
