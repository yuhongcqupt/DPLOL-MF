# Package Description
There are 4 files in “source code” package.
- “DPLOL_MF.py”: simulation for multi-view dataset. Algorithm is DPLOL_MF.
- “DPL_MF.py”: simulation for multi-view dataset. Algorithm is DPL_MF.
- “MvDPL.py”: simulation for multi-view dataset. Algorithm is MvDPL
- “True_DPLOL_MF.py”: Experiment for real mixed frequency data. Algorithm is DPLOL_MF.
- “True_DPL_MF.py”: Experiment for real mixed frequency data. Algorithm is DPL-MF.
- “True_MvDPL.py”: Experiment for real mixed frequency data. Algorithm is MvDPL.<br>

For more details of those algorithms, please refer to “A Novel Discriminative Dictionary Pair Learning Constrained by Ordinal Locality for Mixed Frequency Data Classification”
# Run DPLOL-MF or DPL-MF
When you try to run those 4 files you must set some parameters in main function.
- “trainX”, ”trainY”: train data and train label file path.<br>
*Example*: <br>
&emsp; trainX = ["./stock/day_train.txt", "./stock/month_train.txt"]<br>
&emsp; trainY=["./stock/day_train_la.txt", "./stock/month_train_la.txt"]
- “testX”, ”testY”: test data and test label file path.<br>
*Example*: <br>
&emsp; testX = ["./stock/day_test.txt", "./stock/month_test.txt"]<br>
&emsp; testY=[ "./stock/day_test_la.txt", "./stock/month_test_la.txt"]
- “ratio”: data ratio of each view.<br>
*Example*: <br>
&emsp; For real data: ratio={0:1,1:21}<br>
&emsp; For simulation data: ratio={0:1,1:0.8}<br>

If you get a bad accuracy on your datasets, you should try to adjust hyperparameters referring to our paper.

