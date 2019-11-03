import numpy as np 
import chumpy as ch 

if __name__ == "__main__":
    a = np.array([[1,2], [3,4]])
    b = np.array([[5,6], [7,8]])
    print(np.dot(a, b), type(np.dot(a, b)))
    cha = ch.array(a)
    chb = ch.array(b)
    print(type(cha))
    print(ch.dot(cha, chb), type(ch.dot(cha, chb)))

