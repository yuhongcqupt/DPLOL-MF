# Introduction

![navigation](https://github.com/yuhongcqupt/DPLOL-MF/blob/master/software/img/navigation.png)

<br>

This is navigation window of DPLOL_MF, including some essential information, such as title, paper link address and reference. Paper link address provide the paper that this software based on. We click “RUN DPLOL_MF” button to turn into main window of DPLOL_MF as shown follow.

<br>

![mainwindow](https://github.com/yuhongcqupt/DPLOL-MF/blob/master/software/img/mainwindow.png)

<br>

This window includes 3 parts: parameters setting part, source data inputting part and result shown part. Next we will introduce some details for those 3 parts.

# Set parameter
In the part, we should set last 3 parameters.

- maxIter: maxIter is set to indicate how many times this training takes repeating. Default=30

- dick_size: The number of dictionary atom. Default=18

- H: The number of view or frequency. It is not null.

- nclass: The number of categories. It is not null.

- ratio: There are two data forms. For “MF”, if there are 3 frequency data including daily, monthly and quarterly. Based on those frequency, ratio as “1:21:63”, indicate “day:month:quarter”. That is mean a month including 21 trading days, and one quarter including 63 trading days. For “Simulation”, if there are 3 views data and you want to sampling as “100%:80%:60%” to carry out simulation experiment. Based on the above sampling strategy, ratio as “1:0.8:0.6”.

- MF and Simulation: If the data is real frequency data, you must choose “MF”. If the data you have is multi-view data to carry out simulation experiment, you must choose “Simulation”.

# Input data
In the part, you should open your computer disk to find the input data. There are some rules to limit your input. 
1.	The order of file opening must be: training data file, training data label file, testing data file, testing data label file. 
2.	Each data file corresponds to a data label file.
The input file show as follow:
<br>

![stock](https://github.com/yuhongcqupt/DPLOL-MF/blob/master/software/img/dataset.png)

<br>
There are 3 frequency data. we split it as shown above. We will input this data step by step.
<br>
Step1: input train file. You have to make sure that the order of the files matches the description of the  parameters “ratio”.
<br>

![step1](https://github.com/yuhongcqupt/DPLOL-MF/blob/master/software/img/step1.png)

<br>
Step 2: input train label file.
<br>

![step2](https://github.com/yuhongcqupt/DPLOL-MF/blob/master/software/img/step2.png)

<br>
Step 3: input test file.
<br>

![step3](https://github.com/yuhongcqupt/DPLOL-MF/blob/master/software/img/step3.png)

<br>
Step 4: input test label file.
<br>

![step4](https://github.com/yuhongcqupt/DPLOL-MF/blob/master/software/img/step4.png)


# Run
After you have finished the appeal operation, you can click the button to run the program.
<br>

![waitting](https://github.com/yuhongcqupt/DPLOL-MF/blob/master/software/img/waitting.png)

<br>
After a moment, you can get the result!
<br>

![result](https://github.com/yuhongcqupt/DPLOL-MF/blob/master/software/img/result.png)
