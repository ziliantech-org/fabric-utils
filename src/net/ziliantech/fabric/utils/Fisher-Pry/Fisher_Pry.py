import numpy as np
import pandas as pd
from Fisher_Pry import *
import math
from sklearn import linear_model
import matplotlib.pyplot as plt
import sympy as sp
import os
import openpyxl

def Fisher_Pry_Linear(df, year_max, code, region):
    path = os.getcwd() + '\\img\\' + region + '\\'
    if not os.path.exists(path):
        os.makedirs(path)
    x = np.array(list(df.x))
    y = np.array(list(df.y))
    ind_max = np.array([(y[i + 1] - y[i]) / (x[i + 1] - x[i]) for i in range(len(df) - 1)]).argmax()
    K_est = y[ind_max + 1] * 2

    if K_est < max(df.y):
        K_est = 1.1 * max(df.y)

    x_min = min(df.x)
    year = year_max - x_min

    df.x = df.x - x_min
    df['y_est'] = df['y'] / K_est
    df.y_est = df.apply(lambda x: math.log2(x['y_est'] / (1 - x['y_est'])), axis=1)
    x = np.array(list(df.x)).reshape(-1, 1)
    y_est = np.array(list(df.y_est)).reshape(-1, 1)

    model1 = linear_model.LinearRegression()
    model1.fit(x, y_est)
    b_est = model1.intercept_
    w_est = model1.coef_

    pred_x = np.arange(0, year, 1)
    pred_y_est = w_est * pred_x + b_est
    pred_x.reshape(-1, 1)
    pred_y_est.reshape(-1, 1)
    x = np.array(list(df.x)).reshape(-1, 1)
    x = x + x_min
    pred_x = pred_x + x_min

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(x, y_est)
    plt.plot(pred_x, pred_y_est[0], c='r', label='line')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlim(xmin=x_min)
    plt.xlabel("年")
    plt.ylabel("发展成熟度")
    plt.title(code + "的Fisher-Pry拟合曲线")
    plt.savefig(path + '{}-Fisher-Pry.png'.format(code))

    x = np.array(list(df.x)).reshape(-1, 1)
    y = np.array(list(df.y)).reshape(-1, 1)
    sig_x = np.arange(0, year, 1)
    sig_y = K_est / (1 + np.exp(-1 * (w_est * sig_x + b_est)))

    per_y = 1 / (1 + np.exp(-1 * (w_est * sig_x + b_est)))

    per_y_02 = per_y[per_y > 0.2][0]
    per_y_05 = per_y[per_y > 0.5][0]
    per_y_08 = per_y[per_y > 0.8][0]
    per_y_10 = per_y[per_y > 0.99][0]

    per_x_02 = sig_x[np.where(per_y == per_y_02)[1][0]] + x_min
    per_x_05 = sig_x[np.where(per_y == per_y_05)[1][0]] + x_min
    per_x_08 = sig_x[np.where(per_y == per_y_08)[1][0]] + x_min
    per_x_10 = sig_x[np.where(per_y == per_y_10)[1][0]] + x_min

    per_y_02 = per_y_02 * K_est
    per_y_05 = per_y_05 * K_est
    per_y_08 = per_y_08 * K_est

    sig_x.reshape(-1, 1)
    sig_y.reshape(-1, 1)

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    x = x + x_min
    plt.scatter(x, y)
    sig_x = sig_x + x_min
    plt.xlim(xmin=x_min, xmax=x_min + year)
    plt.ylim(ymin=0, ymax=K_est + K_est * 0.1)
    plt.plot(sig_x, sig_y[0], c='r', label='line')

    plt.scatter(per_x_02, per_y_02, s=50, color='r')
    plt.plot([per_x_02, per_x_02], [0, per_y_02], '--', linewidth=1.5, color='grey')
    plt.annotate("技术萌芽期—%s年" % (per_x_02), xy=(per_x_02, per_y_02), xytext=(per_x_02 + 1, per_y_02))

    plt.scatter(per_x_05, per_y_05, s=50, color='r')
    plt.plot([per_x_05, per_x_05], [0, per_y_05], '--', linewidth=1.5, color='grey')
    plt.annotate("快速成长期—%s年" % (per_x_05), xy=(per_x_05, per_y_05), xytext=(per_x_05 + 1, per_y_05))

    plt.scatter(per_x_08, per_y_08, s=50, color='r')
    plt.plot([per_x_08, per_x_08], [0, per_y_08], '--', linewidth=1.5, color='grey')
    plt.annotate("成熟期—%s年" % (per_x_08), xy=(per_x_08, per_y_08), xytext=(per_x_08 + 1, per_y_08))

    plt.xlabel("年")
    plt.ylabel("数量")
    plt.title(code + "的发展图")
    plt.savefig(path + '{}_logistic.png'.format(code))

    a = w_est[0][0]
    b = b_est[0]
    x = sp.symbols('x')
    func = 1 / (1 + sp.exp(-1 * (a * x + b)))
    fun_diff = sp.diff(func, x)
    diff_x = np.arange(0, year, 1)
    diff_y = []
    for i in diff_x:
        diff_y_tmp = fun_diff.evalf(subs={x: i})
        diff_y.append(100 * diff_y_tmp)
    y_max = max(diff_y)
    x_max = diff_y.index(y_max) + x_min

    diff_y = np.array(diff_y)
    diff_x.reshape(-1, 1)
    diff_y.reshape(-1, 1)
    diff_x = diff_x + x_min

    plt.figure(figsize=(10, 6))
    plt.plot(diff_x, diff_y)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(x_max, y_max, s=50)
    plt.plot([x_max, x_max], [0, y_max], 'k--', linewidth=1.5)
    plt.annotate("%s年" % (x_max), xy=(x_max, y_max), xytext=(x_max + 2, y_max))
    plt.xlim(xmin=x_min)
    plt.ylim(ymin=0)
    plt.ylabel("发展速率%")
    plt.xlabel("年")
    plt.title(code + "的发展速率图")
    plt.savefig(path + '{}-Growth-Rate.png'.format(code))

    data = {'code': [code],
            'year_20': [per_x_02],
            'year_50': [per_x_05],
            'year_80': [per_x_08],
            'K': [K_est],
            'year_final': [per_x_10]}
    res = pd.DataFrame(data)
    return res


