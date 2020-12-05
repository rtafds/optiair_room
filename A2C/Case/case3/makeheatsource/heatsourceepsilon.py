import sys
import re

if __name__ == '__main__':
    before_str="""\
    sideWalls
    {
        type            empty;
    }"""
    after_str="""\
    sideWalls
    {{
        type            empty;
    }}
    {}
    {{
        type            epsilonWallFunction;
        Cmu             0.09;
        kappa           0.41;
        E               9.8;
        value           uniform 0.01;
    }}""".format(sys.argv[2])
    f = open(sys.argv[1],'r')
    body = f.read()
    #print(body.replace(before_str,after_str))
    result = body.replace(before_str,after_str)
    f.close()
    with open(sys.argv[1],'w') as f:
        f.write(result)