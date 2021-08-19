# obscure_object_price_prediction
The purpose of this project is to predict the best possible price for an obscurely shaped, 3D printed object.

The available data consists of various object measurements, such as x, y, and z dimensions, as well as volume, optimal print on the build plate, and several others.  One issue to keep an eye on is that several columns in the data table are nested and have been recoded in a json format, and also that measurements contained in some of these columns are recorded in different units (inches and millimiters). This step required some care in order to correctly extract all the necessary information, and standardize all of the values. 

After the data cleaning step was completed, three different models were tested out and compared for optimal performance. 
