import sys, os
sys.path.append(os.path.dirname(__file__))

from explain import get_explanation

print(get_explanation("PNEUMONIA"))
print(get_explanation("NORMAL"))
