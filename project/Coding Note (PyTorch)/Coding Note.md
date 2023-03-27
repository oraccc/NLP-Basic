### pandas 数据处理



`Series.values`

> Return Series as ndarray or ndarray-like depending on the dtype.
>
> Returns: `numpy.ndarray` or ndarray-like



`pandas.DataFrame.value_counts`

> Return a **Series** containing counts of unique rows in the DataFrame.



`pandas.DataFrame.plot`

> Make plots of **Series or DataFrame**.
>
> Uses the backend specified by the option `plotting.backend`. By default, matplotlib is used.



`pandas.concat(objs, axis=0)`

> 可以沿着一条轴将多个 Pandas 对象连接在一起，`axis`: 整数型参数，默认为 0，表示沿着哪个轴进行连接。当 `axis=0` 时，表示在行方向上进行连接；当 `axis=1` 时，表示在列方向上进行连接。



**添加表头的操作**

对于没有表头的DataFrame，可以使用 `names` 参数来添加表头。

- 如果使用 `names` 参数，可以直接指定一个列表来命名每一列，该列表的长度必须与 DataFrame 中的列数相同，同时不需要设置 header=None 参数。

下面是一个示例代码：

```python
import pandas as pd

# 创建一个没有表头的 DataFrame
data = [[1,2,3],[4,5,6],[7,8,9]]
df = pd.DataFrame(data)

# 使用 names 参数来添加表头
df_names = pd.read_csv('my_file.csv', names=['col1', 'col2', 'col3'])
```

