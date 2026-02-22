# The error "TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'"
# occurs at  in predict_q.
# This means  is None.
#  is called with  in loop.
#  is calculated in loop but initialized to None.
# If features is None when update_q is called, it crashes.

# In my loop logic:
# if 'features' not in locals():
#     # calculate features
#
# But I initialized features = None at start of loop.
# So 'features' IS in locals(), but it is None!
# So the check  is false, and it skips calculation!

# Fix: Change  to

import re

with open('Core/swarm_core_v5_5.py', 'r') as f:
    content = f.read()

content = content.replace("if 'features' not in locals():", "if features is None:")

with open('Core/swarm_core_v5_5.py', 'w') as f:
    f.write(content)
