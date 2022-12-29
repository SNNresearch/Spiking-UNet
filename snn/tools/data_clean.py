import numpy as np

# 包装了一个异常值处理的代码，可以调用
def outliers_proc(data, scale=3):
    """
    用于清洗异常值，默认box_plot(scale=3)进行清洗
    param data: 接收pandas数据格式
    param col_name: pandas列名
    param scale: 尺度
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_serier = data_n
    rule, value = box_plot_outliers(data_serier, box_scale=scale)
    index = np.arange(data_serier.shape[0])[rule[0] | rule[1]]

    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    return data_n
