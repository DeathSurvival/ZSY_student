from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit([['中国','男',22],
         ['美国','女',21],
         ['中国','女',20],
         ['日本','女',19]])

'''
    第一列特征分为三种用三个0,1表示，例中国：1,0,0
    第二列特征分为两种用两个0,1表示，例男：0,1
    第三列特征分为四种用四个0,1表示，列22：0,0,0,1
'''
ans = enc.transform([['中国','男',22]]).toarray()  #toarray装换为矩阵格式
print(ans)

'''
    OneHotEncoder(n_values=’auto’,  categorical_features=’all’,  
    dtype=<class ‘numpy.float64’>,  sparse=True,  handle_unknown=’error’)
    n_values=’auto’，表示每个特征使用几维的数值由数据集自动推断，也可手动设置，例：
        n_values = ['中国','美国'，'日本'，'德国']，则可以将在列表中没显示的特征也会算入，中国：1,0,0,0
    categorical_features = 'all'，这个参数指定了对哪些特征进行编码，默认对所有类别都进行编码,也可自行定义，例：
        categorical_features = [0,2] 或者 [True, False, True]
    dtype=<class ‘numpy.float64’> 表示编码数值格式，默认是浮点型。
    sparse=True 表示编码的格式，默认为 True，即为稀疏的格式(显示非0的位置和值)
    handle_unknown=’error’，其值可以指定为 "error" 或者 "ignore"，即如果碰到未知的类别，是返回一个错误还是忽略它。
'''
_enc = OneHotEncoder(sparse=True)
_ans = _enc.fit_transform([['中国','男',22],
         ['美国','女',21],
         ['中国','女',20],
         ['日本','女',19]])
print(_ans)