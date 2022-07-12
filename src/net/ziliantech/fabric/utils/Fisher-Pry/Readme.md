# Readme

Function *Fisher_Pry_Linear* contains four parameters : **df, year_max, code, region**.

df: This is a **dataframe** like the .csv file *data_demo*. Notice that the first column is named "x", and the second column is named "y".

year_max: the maximal predicted year. The type of this parameter is **int**.

code: the category name of data, like "A01" in derwent code. The type of this parameter is **char**.

region: the region name of data, like "China", "USA", or "global". The type of this parameter is **char**.

---

the data process code for "2.1derwent_USA_year_clean.csv" is as follow:

```python
import pandas as pd
df = pd.read_csv('dataset/2.3derwent_USA_year_clean.csv')
drop_ind = list(df[df["year"]<1900].index)
df = df.drop(drop_ind,axis=0)
df.index = range(len(df))
code_list =set(list(df["code"]))
final_res = pd.DataFrame(columns=['code','year_20','year_50','year_80','K','year_final'])
for code in list(code_list):
	df_sub = pd.DataFrame(columns=['code','year','count'])
	df_sub = df[df["code"]==code]
	if len(df_sub) > 10 :
		df_sub.rename(columns={'year':'x','count':'y'},inplace=True)
		df_sub.sort_values("x",inplace=True)
		df_sub.index = range(len(df_sub))
		y_cum = pd.Series(np.cumsum(np.array(list(df_sub.y))))
		df_sub["y"] = y_cum
		df_sub.index = range(len(df_sub))
		res = Fisher_Pry_Linear(df_sub,2050,code,"USA")
		final_res = pd.concat([final_res,res],axis=0)
final_res.to_excel("res/USA.xlsx",index=False)
```

the data process code for "30clusters美国年变化new.csv" is as follow:

```python
import pandas as pd
df = pd.read_csv('dataset/30clusters美国年变化new.csv')
col = list(df.columns)
code_list = col[1:]
final_res = pd.DataFrame(columns=['code','year_20','year_50','year_80','K','year_final'])
for code in code_list:
    df_sub = pd.DataFrame(columns=['year','count'])
    df_sub["year"] = df["year"]
    df_sub["count"] = df[code]
    df_sub.rename(columns={'year':'x','count':'y'},inplace=True)
    df_sub.sort_values("x",inplace=True)
    df_sub.index = range(len(df_sub))
    y_cum = pd.Series(np.cumsum(np.array(list(df_sub.y))))
    df_sub["y"] = y_cum
    df_sub.index = range(len(df_sub))
    res = Fisher_Pry_Linear(df_sub,2050,code,"topic_USA")
    final_res = pd.concat([final_res,res],axis=0)
final_res.to_excel("excel/topic_USA.xlsx",index=False)
```