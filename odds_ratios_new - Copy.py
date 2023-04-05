
#coding: utf-8



    
import pandas as pd
from osgeo import ogr, gdal,osr
import numpy as np 
files = ['age','sbw_2021','Bf','Sw','Sb','min_temp_jan_daymet','soil_reproj','elev']
names = ['age','sbw','bf','sw','sb','mj','st','elev'] 
pred = {}
transformers = []
cols_list = []
rows_list = [] 

for fi,n in zip(files,names): 
    print(fi)
    file_name_raster = fi
    src_ds = gdal.Open('final/'+file_name_raster+'.tif')
    rb1=src_ds.GetRasterBand(1)
    cols = src_ds.RasterXSize
    cols_list.append(cols)
    rows = src_ds.RasterYSize
    rows_list.append(rows) 
    data = rb1.ReadAsArray(0, 0, cols, rows)
    print('Success in reading file.........................................') 
    pred[n] = data.flatten()
    print(len(data.flatten()))
    transform=src_ds.GetGeoTransform()
    transformers.append(transform)

pred['age'] = pred['age'] + (2021-2011)
col_num = cols_list[0]
row_num = rows_list[0]
ulx, xres, xskew, uly, yskew, yres  = transformers[0]
lrx = ulx + (col_num * xres)
lry = uly + (row_num * yres)


Yi = np.linspace(np.min([uly,lry]), np.max([uly,lry]), row_num)
Xi = np.linspace(np.min([ulx,lrx]), np.max([ulx,lrx]), col_num)


Xi, Yi = np.meshgrid(Xi, Yi)
Xi, Yi = Xi.flatten(), Yi.flatten()

X_reshape = Xi.reshape(row_num,col_num)[::-1]
Xi = X_reshape.flatten()
Y_reshape = Yi.reshape(row_num,col_num)[::-1]
Yi = Y_reshape.flatten()


pred['lon'] = Xi
pred['lat'] = Yi


df = pd.DataFrame(pred).dropna(how='any')

for nam in names: 

    df = df[df[nam] != -3.4028234663852886e+38]
    df = df[df[nam] != -9999]
    
print(df.head(n=50))

df = df.sample(n=700000,random_state=42)

# Bin it

max_bf = df['bf'].max()
df['category_bf'] = np.select(
    [
        (df['bf'] >= 0) & (df['bf'] < 10),
        (df['bf'] >= 10) & (df['bf'] < 20),
        (df['bf'] >= 20) & (df['bf'] < 30),
        (df['bf'] >= 30) & (df['bf'] <= 40)
    ], 
    [
        '1-10',
        '10-20',
        '20-30',
        '30-40'
    ]
)

# Check for empty bins
print(pd.value_counts(df['category_bf']))

# Sb
max_sb = df['sb'].max()
df['category_sb'] = np.select(
    [
        (df['sb'] >= 0) & (df['sb'] < 10),
        (df['sb'] >= 10) & (df['sb'] < 20),
        (df['sb'] >= 20) & (df['sb'] < 30),
        (df['sb'] >= 30) & (df['sb'] < 40),
        (df['sb'] >= 40) & (df['sb'] < 50),
        (df['sb'] >= 50) & (df['sb'] < 60),
        (df['sb'] >= 60) & (df['sb'] < 70),
        (df['sb'] >= 70) & (df['sb'] < 80),
        (df['sb'] >= 80) & (df['sb'] < 90),
        (df['sb'] >= 90) & (df['sb'] <= 100)
    ], 
    [
        '1-10',
        '10-20',
        '20-30',
        '30-40',
        '40-50',
        '50-60',
        '60-70',
        '70-80',
        '80-90',
        '90-100'
    ]
)

# Check for empty bins
print(pd.value_counts(df['category_sb']))

# Sw
max_sw = df['sw'].max()
df['category_sw'] = np.select(
    [
        (df['sw'] >= 0) & (df['sw'] < 10),
        (df['sw'] >= 10) & (df['sw'] < 20),
        (df['sw'] >= 20) & (df['sw'] <= 30)
    ], 
    [
        '1-10',
        '10-20',
        '20-30'
    ]
)

# Check for empty bins
print(pd.value_counts(df['category_sw']))

# Elevation 
max_elev = int(np.ceil(df['elev'].max()))
df['category_elev'] = pd.cut(df['elev'], 
                             bins=np.arange(0, max_elev+50, 50), 
                             labels=[f"{i+1}-{i+50}" for i in range(0, max_elev, 50)])

# Check for empty bins
print(pd.value_counts(df['category_elev']))
import statsmodels.api as sm
df['quantile_elev'] = pd.qcut(df['elev'], q=5, labels=['elev1','elev2','elev3','elev4','elev5'])
print(pd.value_counts(df['quantile_elev']))

max_jan = np.ceil(df['mj'].max()) # 0
min_jan = np.floor(df['mj'].min()) # ~-35

df['category_mj'] = np.select(
[
(df['mj'] >= -35) & (df['mj'] < -30),
(df['mj'] >= -30) & (df['mj'] < -25),
(df['mj'] >= -25) & (df['mj'] < -20),
(df['mj'] >= -20) & (df['mj'] < -15),
(df['mj'] >= -15) & (df['mj'] < -10),
(df['mj'] >= -10) & (df['mj'] < -5),
(df['mj'] >= -5) & (df['mj'] <= 0)
],
[
'-35--30',
'-30--25',
'-25--20',
'-20--15',
'-15--10',
'-10--5',
'-5-0'
],
default=''
)

df['quantile_mj'] = pd.qcut(df['mj'], q=12, labels=['mj1','mj2','mj3','mj4','mj5','mj6','mj7'\
                                                    ,'mj8','mj9','mj10','mj11','mj12'])
print(pd.value_counts(df['quantile_mj']))

pd.crosstab(index=df['category_mj'], columns='count')

unique_st = df['st'].unique()

df['category_st'] = np.select(
[
df['st'] == 7,
df['st'] == 0,
df['st'] == 4,
df['st'] == 6,
df['st'] == 9,
df['st'] == 1
],
[
'7',
'0',
'4',
'6',
'9',
'1'
],
default=''
)

pd.crosstab(index=df['category_st'], columns='count')

#SBW

df['category_sbw'] = np.select(
[
df['sbw'] == 1.0,
df['sbw'] == 0.0
],
[
'1',
'0'
],
default=''
)

pd.crosstab(index=df['category_sbw'], columns='count')

#Model
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from patsy import dmatrix

# Create a LabelEncoder object for each categorical feature
encoders = {}
for feature in ['category_sbw','category_elev', 'category_bf', 'category_sw', 'category_sb', 'category_mj', 'category_st']:
    encoders[feature] = LabelEncoder()
    df[feature] = encoders[feature].fit_transform(df[feature])


dummy_df_elev = pd.get_dummies(df['category_elev'], prefix='category_elev')
dummy_df_bf = pd.get_dummies(df['category_bf'], prefix='category_bf')
dummy_df_sw = pd.get_dummies(df['category_sw'], prefix='category_sw')
dummy_df_sb = pd.get_dummies(df['category_sb'], prefix='category_sb')
dummy_df_mj = pd.get_dummies(df['category_mj'], prefix='category_mj')
dummy_df_st = pd.get_dummies(df['category_st'], prefix='category_st')
#X_spline = dmatrix(formula, data=df, return_type='dataframe')
#X_spline = dmatrix(formula, data=df, return_type='dataframe')
# Combine the dummy variables with the other independent variables
X = pd.concat([df['category_sbw'],dummy_df_elev,dummy_df_bf,dummy_df_sw,dummy_df_sb,dummy_df_st,df['lat'],df['lon']], axis=1)
y = np.asarray(df['category_sbw']).astype('int')

# Fit the logistic regression model
import statsmodels.formula.api as smf
#model = smf.logit("category_sbw ~ C(quantile_elev,Treatment(reference=1))\
#+C(category_bf,Treatment(reference=0))+C(category_sw,Treatment(reference=0))+\
#C(category_sb,Treatment(reference=0))+C(category_st,Treatment(reference=4))+\
#bs(lat,df=3) + bs(lon,df=3)",data=df)

model = smf.logit("category_sbw ~ C(quantile_elev,Treatment(reference=1))\
+C(category_bf,Treatment(reference=0))+C(category_sw,Treatment(reference=0))+lat*lon",data=df)

log_reg = model.fit()

print(log_reg.summary())

pd.options.display.float_format = '{:.2f}'.format
odds_ratios = pd.DataFrame(
    {
        "OR": np.exp(log_reg.params),
        "Lower CI": np.exp(log_reg.conf_int()[0]),
        "Upper CI": np.exp(log_reg.conf_int()[1]),
    }
)
print(odds_ratios)
odds_ratios = odds_ratios.drop(odds_ratios.index[0])
print(odds_ratios)
#Compute R-squared value
r2_value = [log_reg.prsquared]
print(r2_value)
