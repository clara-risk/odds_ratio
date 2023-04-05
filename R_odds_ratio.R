#Python block 
setwd("C:/Users/clara/Documents/proposal_2023/odds_ratio")
Sys.setenv(RETICULATE_PYTHON = "C:/Python38/")
library(reticulate)

result <- py_run_string("import pandas as pd
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

df = df.sample(n=10000,random_state=42)")

# R code

library(classInt)
library(dplyr)
library(sp)

df <- py$df

df <- df
print(df)
# Convert the SpatialPointsDataFrame to a data frame
df <- data.frame(df)

# Bf

# By 10 
max_bf <- max(df$bf) # 40

df <- df %>%
  mutate(category_bf = case_when(
    bf >= 0 & bf < 10 ~ "1-10",
    bf >= 10 & bf < 20 ~ "10-20",
    bf >= 20 & bf < 30 ~ "20-30",
    bf >= 30 & bf <= 40 ~ "30-40",
  ))

# Jenks Natural Breaks 
breaks <- classIntervals(df$bf, n = 4, style = "jenks")$brks
df$category_jenksBf <- cut(df$bf, breaks, include.lowest = TRUE)

# Check for empty bins
table(df$category_bf)

#Sb 

max_sb <- max(df$sb) # 100

df <- df %>%
  mutate(category_sb = case_when(
    sb >= 0 & sb < 10 ~ "1-10",
    sb >= 10 & sb < 20 ~ "10-20",
    sb >= 20 & sb < 30 ~ "20-30",
    sb >= 30 & sb < 40 ~ "30-40",
    sb >= 40 & sb < 50 ~ "40-50",
    sb >= 50 & sb < 60 ~ "50-60",
    sb >= 60 & sb < 70 ~ "60-70",
    sb >= 70 & sb < 80 ~ "70-80",
    sb >= 80 & sb < 90 ~ "80-90",
    sb >= 90 & sb <= 100 ~ "90-100"
  ))

# Check for empty bins
table(df$category_sb)

#Sw

max_sw <- max(df$sw) # 21

df <- df %>%
  mutate(category_sw = case_when(
    sw >= 0 & sw < 10 ~ "1-10",
    sw >= 10 & sw < 20 ~ "10-20",
    sw >= 20 & sw <= 30 ~ "20-30",
  ))

# Check for empty bins
table(df$category_sw)

# Elevation 

max_elev <- max(df$elev) # ~600 (538 in sample)

max_elev <- max(df$elev)
df <- df %>%
  mutate(category_elev = cut(elev, breaks = seq(0, max_elev, by = 50), 
                             labels = paste(seq(1, max_elev/50) * 50 - 49, 
                                            seq(1, max_elev/50) * 50, sep = "-")))

table(df$category_elev)


# Min Jan 

max_jan <- ceiling(max(df$mj)) # 0
min_jan <- floor(min(df$mj)) # ~-35

df <- df %>%
  mutate(category_mj = case_when(
    mj >= -35 & mj < -30 ~ "-35--30",
    mj >= -30 & mj < -25 ~ "-30--25",
    mj >= -25 & mj < -20  ~ "-25--20",
    mj >= -20  & mj < -15 ~ "-20--15",
    mj >= -15   & mj < -10  ~ "-15--10",
    mj >= -10  & mj < -5  ~ "-10--5",
    mj >= -5  & mj <= 0  ~ "-5-0"
  ))

table(df$category_mj)

# st

unique_st <- unique(df$st)

df <- df %>%
  mutate(category_st = case_when(
    st == 7 ~ "7",
    st == 0 ~ "0",
    st == 4 ~ "4",
    st == 6 ~ "6",
    st == 9 ~ "9",
    st == 1 ~ "1",
  ))

table(df$category_st)

# sbw

df <- df %>%
  mutate(category_sbw = case_when(
    sbw == 1.0 ~ 1,
    sbw == 0.0 ~ 0,
  ))

table(df$category_sbw)


library(nlme)
#install.packages('splines')
library(splines)
#Omit the bins without data --> is this valid? 

df <- na.omit(df)

#Now we will set the referent group 
df$category_elev <- relevel(df$category_elev, ref = "251-300")


#The b splines are resulting in overfitting? 

#fit1 <- glm(category_sbw ~ category_elev + category_bf + category_sw + category_sb + category_mj + category_st+ bs(lat, df = 200)*bs(lon, df = 200), data = df, family = binomial(link = "logit"))
#AIC_model1 <- AIC(fit1)
#Should recreate the SBW maps With just lat lon others constant to check if working properly and recreating corEXP
#print(AIC_model1)

#glmm
#fit2 <- gls(category_sbw ~ category_elev + category_bf + category_sw + category_sb + 
              #category_mj + category_st, data = df, correlation = corExp(form = ~lon + lat, nugget = FALSE),
            #method = "REML")

#AIC_model2 <- AIC(fit2)

#print(AIC_model2)

# Fit the model using the spherical correlation structure

fit1 <- gls(category_sbw ~ category_elev + category_bf + category_sw + category_sb + 
              category_mj + category_st, data = df, correlation = corSpher(form = ~lon + lat, nugget = TRUE),
            method = "REML")


# Unpenalized model 
# Extract coefficients from model
coef_table <- data.frame(coef(fit1))

SE <- summary(fit1)$coefficients[, "Std. Error"]
p_values <- summary(fit1)$coefficients[, "Pr(>|z|)"]
#est <- summary(fit1)$coefficients[, "Estimates"]

# Calculate odds ratios and confidence intervals using natural number e
coef_table$odds_ratio <- exp(coef_table[,1])
coef_table$lower <- exp(est - qnorm(0.975) * SE)
coef_table$upper <- exp(coef_table[,1] + qnorm(0.975) * SE)
coef_table$p_value <- p_values


# Combine the two tables
important_info <- data.frame(coef_table[,1], coef_table$odds_ratio, 
                             summary_table[,1], coef_table$lower, coef_table$upper)
colnames(important_info) <- c("Coefficient", "Odds Ratio", "p-value", "LowerCI", "UpperCI")

# Print table
important_info

