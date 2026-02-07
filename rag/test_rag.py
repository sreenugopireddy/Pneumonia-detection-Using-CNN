import sys, os
sys.path.append(os.path.dirname(__file__))

from explain import rag_answer

q = "What are the symptoms and treatment of pneumonia?"

print("\nQUERY:\n", q)
print("\nANSWER:\n")
print(rag_answer(q))
