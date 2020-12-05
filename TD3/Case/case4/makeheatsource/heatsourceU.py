import sys
import re
## 第一引数に対象のファイル、第二引数に名前
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
        type            noSlip;
    }}""".format(sys.argv[2])
    f = open(sys.argv[1],'r')
    body = f.read()
    #print(body.replace(before_str,after_str))
    result = body.replace(before_str,after_str)
    f.close()
    with open(sys.argv[1],'w') as f:
        f.write(result)