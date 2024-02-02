# Plot the results from the data frame is contained in the file
import matplotlib.pyplot as plt
import pandas as pd


# file name containing the data frame
data_file_name = 'epochs_aver_df.csv'


df = pd.read_csv(data_file_name)
df.plot( x = 'Epoch', y = ['epoch_trn_loss', 'epoch_val_loss'], figsize=(20,10), grid=True)
plt.legend(["trn_loss", "val_loss"]);
plt.show() 