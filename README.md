# Traffic Prediction

datasets
1. citibikenyc-data
num_station: 331

##HA: 
in/out l2-loss: 3.3003/3.1226
in/out rmse: 2.7905/2.6688
in/out rmlse: 0.5246/0.5287
in/out er: 0.7437/0.7435

##VAR: 
in/out l2-loss: 3.2899/3.0597
in/out rmse: 2.8890/2.6994
in/out rmlse: 0.6004/0.5904
in/out er: 0.9238/0.8629

##ARMA:
in/out l2-loss: 4.1018/
in/out rmse: 3.4408/
in/out rmlse: 0.6595/
in/out er: 2.0901/

##HP-MSI:
check-out l2-loss: 3.4588
check-out rmse: 2.8921
check-out rmlse: 0.5611
check-out er: 0.6771

----- check-in inference algorithm 1 (same as check-out prediction) ------
check-out l2-loss: 3.2073
check-out rmse: 2.7482
check-out rmlse: 0.5613
check-out er: 0.6835

----- check-in inference algorithm 2 -----
check-in l2-loss: 5.8643
check-in rmlse: 0.9132
when using real check-out data,
check-in l2-loss: 5.9432
check-in rmlse: 0.9297


