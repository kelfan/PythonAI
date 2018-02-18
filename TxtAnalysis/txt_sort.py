import codecs
import re


def read_file_utf8(filename):
    """
    write to file in utf8 format
    :param data:
    :param filename:
    :return:
    """
    file = codecs.open(filename, "rb", "utf-8")
    data = file.read()
    file.close()
    print("read " + filename + " success")
    return data


def write_file_utf8(filename, data):
    """
    write to file in utf8 format
    :param data:
    :param filename:
    :return:
    """
    file = codecs.open(filename, "wb", "utf-8")
    file.write(data)
    file.close()
    print("write " + filename + " success")
    return 1


txt = read_file_utf8("./a.txt")
txt_list = txt.split("  ??Vol ")
txt_list.sort()
new_dict = dict()
for line in txt_list:
    m = re.search('Chapter ([0-9]+)', line)
    if m:
        key = m.group(1)
        key = key.zfill(4)
        new_dict[key] = line
# for line in txt_list:
#     if ("\r\n\r\n" != line) & ('\ufeff\r\n\r\n'!=line):
#         m = re.search('Chapter ([0-9]+)', line)
#         key = int(m.group(1))
#         log_worker.log_warn(txt_list.index(line))
#         log_worker.log_info(line)
#         new_dict[key] = line
sort_tuple = [(k, new_dict[k]) for k in sorted(new_dict.keys())]
text = ""
for line in sort_tuple:
    text = text + line[1]
write_file_utf8("./c.txt",text)
end = 0
