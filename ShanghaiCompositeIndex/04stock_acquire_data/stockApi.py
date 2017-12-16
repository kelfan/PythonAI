"""
@tushare: https://pypi.python.org/pypi/tushare#downloads
@source: https://github.com/dengdaiyemanren/python/wiki/%E8%8E%B7%E5%8F%96%E4%BA%A4%E6%98%93%E6%95%B0%E6%8D%AEAPI%E6%B1%87%E6%80%BB

"""
import os
import tushare as ts
import pandas
import platform
from datetime import date
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime

stockcode = '300201'
filename = stockcode+'.csv'
file = Path(filename)
data = []
if file.is_file():
    data = pandas.read_csv(filename,parse_dates = True,
                  index_col=0)
    if date.fromtimestamp(creation_date(file)) != date.today(): # the date of records is not
        data = ts.get_hist_data(stockcode)  # 一次性获取全部日k线数据
        data.to_csv(filename)
else:
    data = ts.get_hist_data(stockcode)  # 一次性获取全部日k线数据
    data.to_csv(filename)

datetime = pandas.to_datetime(data.index)
weekday = datetime.weekday.values
monthDay = datetime.day.values
data['weekday'] = weekday
data['monthDay'] = monthDay
lb_make = LabelEncoder()
