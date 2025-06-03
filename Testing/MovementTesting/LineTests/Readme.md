# MoveTest1 
 - line 1 goes top to bottom through the center mark
 - line 2 goes right to left through the center mark
# MoveTest2
 - line 3 goes from antenna 2 to antenna 4 through the center mark
 - line 4 goes from antenna 3 to antenna 1 through the center mark


# LineMetadata.csv 
Contains information about the line path being tested for each csv file contained in the MoveTest# Folders.
Details concerning this data are as follows:
 - Path # is defined by the points contained by the line (not the order in which the points are collected -> i.e. right to left vs left to right for a straight line with the same start and end points is logged as the same line #)
 - Speed is loosely defined. Slow was Donald using a step per 1'x1' block while walking, and fast was Donald using a step per 2 1'x1' blocks while walking.
 - Orientation 1, 2, 3, 4 refer to the tags faces as it rotates 360 degrees around a ground to ceiling axis:
   - 1 -> tag is parallel to the vector between antennas 1 and 2 (or 3 and 4)
   - 2 -> tag is parallel to the vector between antennas 2 and 3 (or 1 and 4)
   - 3 -> tag is parallel to the face of antenna 3 (or 1)
   - 4 -> tag is parallel to the face of antenna 4 (or 2)
  
   

