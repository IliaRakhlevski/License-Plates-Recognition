# Convert the data received from the VGG Image Annotator 
# to more appropriate format is used for training of the model
import pandas as pd
import ast
import numpy as np


# image sizes
ROWS = 500
COLS = 900

# create new empty data frame
df_new = pd.DataFrame(columns = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"])

df_orig = pd.read_csv('df_orig.csv')

for ix in range(df_orig.shape[0]):
    filename = df_orig['filename'][ix]
    region_shape_attributes =  df_orig['region_shape_attributes'][ix]
    region_attributes = df_orig['region_attributes'][ix]

    d_region_shape_attributes = ast.literal_eval(region_shape_attributes)
    d_region_attributes = ast.literal_eval(region_attributes)

    # create new box for a digit
    figure_class = str(d_region_attributes['class'])
    col_st = np.float(d_region_shape_attributes['x'])
    row_st = np.float(d_region_shape_attributes['y'])
    width = np.float(d_region_shape_attributes['width'])
    height = np.float(d_region_shape_attributes['height'])
    
    col_end = col_st + width
    row_end = row_st + height
    
    # create new row
    new_row = [{"ImageID":filename, "LabelName":figure_class, "XMin":(col_st / COLS),
                "XMax":(col_end / COLS), "YMin":(row_st / ROWS), "YMax":(row_end / ROWS)}]
    
    # add a new row to the new data frame
    df_new = df_new.append(new_row, ignore_index=True)
    
# save the new created data frame
df_new.to_csv('df_new.csv', index=False)
