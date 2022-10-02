import pandas as pd
data1 = pd.read_csv('知乎评论.csv')
def is_chinese(uchar):
    if uchar >= '\u4e00' and uchar <= '\u9fa5':
        return True
    else:
        return False


def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str

