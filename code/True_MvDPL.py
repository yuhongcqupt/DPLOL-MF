import numpy as np
import pandas as pd
from numpy import linalg as LA
from functools import reduce
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import time

class True_MvDPL:
    def __init__(self,maxIter,H,nclass,dict_size,trainX,trainY,testX,testY,ratio):
        self.maxIter = maxIter  # The maximum number of iterations
        self.H = H  # 视图数
        self.nclass = nclass  # 类簇数
        self.dict_size = dict_size  # 初始化字典的原子数
        # 参数
        self.tau = 0.1
        self.lambda1 = 0.1
        self.gamma = 0.0001
        self.beta = 0.001
        self.citeT = 1e-6

        self.datafile = trainX
        self.labelfile = trainY
        self.testfile = testX
        self.true_testlabel_file = testY
        self.samp_freq = ratio


    def loaddata(self,datafile):
        return np.array(pd.read_csv(datafile, sep="\t", header=None)).astype(np.float)


    # 数据处理函数（列和为1）
    def normcol_equal(self,matin):
        # matout = argmin||matout-matin||_F^2, s.t. matout(:,i)=1
        matout = matin / np.tile(np.sqrt(np.sum(np.multiply(matin, matin), axis=0) + np.spacing(1)),
                                 [np.shape(matin)[0], 1])
        return matout


    # 处理列和小于1的函数
    def normcol_lessequal(self,matin):
        part = np.sqrt(np.sum(np.multiply(matin, matin), axis=0) + np.spacing(1))
        part[part < 1] = 1
        matout = matin / np.tile(part, (np.shape(matin)[0], 1))
        return matout


    def other(self,id, c):
        other_id = []
        for c1 in range(nclass):
            if c1 != c:
                other_id.extend(id[c])
        return other_id

    # 按列拼接
    def contacth(self,a, b):
        return np.hstack((a, b))
    # 按行拼接
    def contactv(self,a, b):
        return np.vstack((a, b))

    def pca(self,XMat,dim):
        # sNum, dNum = np.shape(XMat)
        dNum, sNum = np.shape(XMat)
        average = np.mean(XMat,axis=1)  # 按列求均值
        # 如果样本数大于特征数，正常计算
        if sNum > dNum or sNum == dNum:
            avgs = np.tile(average, (sNum, 1)).T  # 4*10
            data_adjust = (XMat - avgs)
            covX = np.dot(data_adjust,data_adjust.T)/(sNum-1)
            # covX = np.cov(data_adjust)  # 计算协方差矩阵 dnum*dum
            featValue, featVec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
            index = np.argsort(-featValue)  # 依照featValue进行从大到小排序
            k = dim
            selectVec = np.mat(featVec.T[index[:k]].T)
            dimMatrix = selectVec
            return dimMatrix
        # 如果样本数小于特征数
        else:
            avgs = np.tile(average, (sNum, 1)).T  # 4*3
            # data_adjust = (XMat - avgs)/(sNum-1)
            data_adjust = (XMat - avgs)
            # covX = np.cov(data_adjust.T)  # 计算协方差矩阵 snum*snum 3*3
            covX = np.dot(data_adjust.T, data_adjust) / (sNum - 1)
            featValue, featVec = LA.eig(covX)  # 求解协方差矩阵的特征值和特征向量
            index = np.argsort(-featValue)  # 依照featValue进行从大到小排序
            k = dim  # 或者用样本数
            dimMatrix = np.zeros([dNum, k])
            for i in range(k):
                dimMatrix[:, i] = np.dot(data_adjust, featVec[:, index[i]]) / (np.sqrt(featValue[index[i]]))
            return np.mat(dimMatrix)

    def initilization(self,MSData, id, H, dict_size):
        # 对不同视图的数据，初始化D,P,X
        # dim = np.shape(MSData[0])[1]  # 样本的特征维度
        # I_mat1 = np.eye(dim)  # 单位矩阵
        classA = {}
        classD = {}
        classP = {}
        classX = {}
        dataInvmat = {}
        classData = {}
        contact_D = {}
        I_mat2 = np.eye(dict_size)
        for h in range(H):
            classA[h] = {}
            classD[h] = {}
            classP[h] = {}
            classX[h] = {}
            dataInvmat[h] = {}
            classData[h] = {}

            dim = np.shape(MSData[h])[1]  # 样本的特征维度
            I_mat1 = np.eye(dim)  # 单位矩阵

            for c in range(nclass):
                temData = MSData[h].T[:, id[h][c]]
                classData[h][c] = temData
                classA[h][c] = temData
                classD[h][c] = self.normcol_equal(np.random.randn(dim, dict_size))  # 47 * 30
                classP[h][c] = self.normcol_equal(np.random.randn(dim, dict_size)).T
                inv_mat = np.linalg.inv(np.dot(classD[h][c].T, classD[h][c]) + self.tau * I_mat2)
                # print(classD[h][c])
                classX[h][c] = np.dot(inv_mat, (np.dot(classD[h][c].T, temData) + self.tau * np.dot(classP[h][c], temData)))
                # classX[h][c] = update_X()

                # 其他类簇数据矩阵
                id_ = self.other(id[h], c)
                temDataC = MSData[h].T[:, id_]
                dataInvmat[h][c] = np.linalg.inv(
                    self.tau * np.dot(temData, temData.T) + self.lambda1 * np.dot(temDataC, temDataC.T) + self.gamma * I_mat1)
            contact_D[h] = reduce(self.contacth, [classD[h][c] for c in range(nclass)])
        return classD, classP, classX, dataInvmat, classData, contact_D


    def update_X(self,D, datamat, P):
        X = {}
        I_mat = np.eye(self.dict_size)
        for c in range(nclass):
            tempDict = D[c]
            temData = datamat[c]
            inv_mat = np.linalg.inv((np.dot(tempDict.T, tempDict) + self.tau * I_mat))
            X[c] = np.dot(inv_mat, (np.dot(tempDict.T, temData) + self.tau * np.dot(P[c], temData)))
        return X


    def update_P(self,X, dataInv, P, datamat):
        for c in range(nclass):
            P[c] = np.dot(self.tau * np.dot(X[c], datamat[c].T), dataInv[c])
        return P


    # def update_D(X, datamat, D, hk):
    def update_D(self,X, datamat, D):
        # 类簇系数矩阵X[1]的行
        I_mat = np.eye(np.shape(X[1])[0])
        for c in range(nclass):
            tempX = X[c]
            tempData = datamat[c]
            rho = 1
            rate_rho = 1.2
            tempS = D[c]
            tempT = np.zeros(np.shape(tempS))
            previousD = D[c]
            iter = 1
            error = 1
            while error > 1e-8 and iter < 100:
                # inv = np.linalg.inv((beta * hk + rho) * I_mat + np.dot(tempX, tempX.T))
                inv = np.linalg.inv(rho * I_mat + np.dot(tempX, tempX.T))
                # print(np.shape(np.dot(tempData, tempX.T)))
                # print(np.shape(rho * (tempS - tempT)))
                part2 = np.dot(tempData, tempX.T) + rho * (tempS - tempT)
                tempD = np.dot(part2,inv)
                tempS = self.normcol_lessequal(tempD + tempT)
                tempT = tempT + tempD - tempS
                rho = rate_rho * rho
                # print(type(previousD - tempD))
                error = np.mean(np.mean(np.array(previousD - tempD) ** 2, axis=0))
                previousD = tempD
                iter += 1
            D[c] = tempD
        # contact_D_h = reduce(contacth, [D[c] for c in range(nclass)])
        # return D,contact_D_h
        return D

    def all_energy(self,MSData,D,datamat,contact_D,P,X,id):
        error = 0
        for h in range(H):
            # # 计算其他视图的字典冗余量
            # hk = 0
            # for l in range(H):
            #     if l != h:
            #         hk += np.linalg.norm(contact_D[l], 'fro')
            gap_class = 0
            for c in range(nclass):
                # 其他类簇数据矩阵
                id_ = self.other(id[h], c)
                temDataC = MSData[h].T[:, id_]
                gap1 = LA.norm(datamat[h][c]-np.dot(D[h][c],X[h][c]),'fro')
                gap2 = LA.norm(np.dot(P[h][c],datamat[h][c])-X[h][c],'fro')
                gap3 = LA.norm(np.dot(P[h][c],temDataC),'fro')
                gap4 = LA.norm(D[h][c],'fro')
                # gap_class += gap1+tau*gap2+lambda1*gap3+beta*hk*gap4
                gap_class += gap1 + self.tau * gap2 + self.lambda1 * gap3
            error += gap_class
        return error

    ## 对比实验 DPL MSADL 区别：MSADL 有字典间冗余项
    def MSD_SADL(self,MSData, id, H):
        # 初始化综合字典 D和分析字典 P 同时利用更新公式得到系数矩阵 X
        D, P, X, dataInv, datamat, contact_D = self.initilization(MSData, id, H, self.dict_size)
        # pref_error = all_energy(D,datamat,contact_D,P,X,id)
        ms_iter = 1
        while ms_iter < self.maxIter:
            # 更新分析字典矩阵
            for h in range(H):
                P[h] = self.update_P(X[h], dataInv[h], P[h], datamat[h])
            # 更新系数矩阵
            for h in range(H):
                X[h] = self.update_X(D[h], datamat[h], P[h])
            # 更新综合字典矩阵
            for h in range(H):
                D[h] = self.update_D(X[h], datamat[h], D[h])
            ms_iter += 1
        return D, P

    def code_num(self,z, f):
        code_z = {}
        code_z[0] = {}
        for i in f[0]:
            code_z[0][i] = z[0][:, i]
        for k in range(1, len(z)):
            code_z[k] = {}
            and_z_f = list(set(f[0]).intersection(set(f[k])))
            for i in and_z_f:
                code_z[k][i] = z[k][:, and_z_f.index(i)]
        return code_z

    # 采样分类方法
    def classification_ADL(self,test_data, D, P, samp_freq):
        sample_num = np.shape(test_data[0])[1]
        a = list(range(sample_num))
        f = {}
        for h in range(H):
            b = []
            b.extend(a[0:sample_num:samp_freq[h]])
            f[h] = b
        print(np.array(f[1]).shape)
        f[2] = f[2][:-1]

        code_z = self.code_num(test_data, f)

        sample_h = {}
        # 生成样本--样本对应哪个视图字典
        for i in range(sample_num):
            sample_h[i] = []
            for h in f.keys():
                if i in f[h]:
                    sample_h[i].append(h)
                else:
                    continue

        tt_start = time.clock()  # 计算打标时间
        test_label = []
        for i in range(sample_num):
            ert_ci = {}
            for c in range(nclass):
                ert_ci[c] = 0
                for h in sample_h[i]:
                    M = np.dot(D[h][c], P[h][c])
                    ert_ci[c] += np.sum(np.abs(np.dot(M, code_z[h][i]) - code_z[h][i]), axis=0)
            test_label.append(min(ert_ci, key=ert_ci.get))
        tt_time = (time.clock() - tt_start)
        print("测试打标时间：", tt_time)
        return test_label

    # 采样算法
    def sample_alg(self,data, label, samp_fre):
        H = len(data)
        data_new = {}
        label_new = {}
        label_set = set(label[0])
        for h in range(H):
            label_new[h] = []
            sep = int(samp_fre[h] * 100)  #
            l = len(data[h])
            data_new[h] = reduce(self.contactv, [data[h][i * 100:i * 100 + sep, :] for i in range(l // 100)])
            for i in range(l // 100):
                label_new[h].extend(label[h][i * 100:i * 100 + sep])
            if set(label_new[h]) != label_set:
                print("请更换采样策略...")  # 10取2  100取20 ...
        return data_new, label_new

    def run(self):

        MSData = {}
        z = {}
        label = {}
        h_dimension = {}
        true_testlabel = {}
        for h in range(H):
            MSData[h] = self.loaddata(self.datafile[h])
            z[h] = self.loaddata(self.testfile[h])
            h_dimension[h] = np.shape(MSData[h])[1]
            label[h] = list(np.array(pd.read_csv(self.labelfile[h], sep="\t", header=None)).astype(np.int)[0, :])
            true_testlabel[h] = list(
                np.array(pd.read_csv(self.true_testlabel_file[h], sep="\t", header=None)).astype(np.int)[0, :])

        # 转换成非负 降维 归一化0-1
        scaler = MinMaxScaler()

        # 训练 测试数据-标准化（standardization）
        for h in range(H):
            # scaler.fit(MSData[h])
            MSData[h] = scaler.fit_transform(MSData[h])
            scaler.fit(z[h])
            z[h] = scaler.transform(z[h]).T

        # 类簇索引列表-方便学习类簇数据
        id = {}
        for h in range(H):
            id[h] = {}
            for c in range(nclass):
                id[h][c] = [i for i, x in enumerate(label[h]) if x == c]

        acc_list = []
        f1_score_list = []
        for i in range(10):
            # 训练学习
            tr_start = time.clock()
            D, P = self.MSD_SADL(MSData, id, H)
            tr_time = (time.clock() - tr_start)
            # print(D, P)

            # 测试样本打标
            tt_start = time.clock()
            test_label = self.classification_ADL(z, D, P, self.samp_freq)
            tt_time = (time.clock() - tt_start)
            acc_list.append(accuracy_score(true_testlabel[0], test_label))
            f1_score_list.append(metrics.f1_score(true_testlabel[0], test_label, average='weighted'))
            # print(test_label)
            print("acc:", acc_list[i])
            # print("recall:", metrics.recall_score(true_testlabel[0], test_label, average='micro'))
            print("f1_score:", f1_score_list[i])

            print("train time used:", tr_time)
            print("test time used:", tt_time)
        avg_acc = np.average(acc_list)
        avg_f1 = np.average(f1_score_list)
        std_acc = np.std(acc_list)
        std_f1 = np.std(f1_score_list)
        print(f"std_acc:{std_acc}")
        print(f"std_f1:{std_f1}")
        print(f"max_acc:{max(acc_list)}")
        print(f"max_f1:{max(f1_score_list)}")
        print(f"avarge_acc:{avg_acc}")
        print(f"avarge_f1:{avg_f1}")
        print(f"min_acc:{min(acc_list)}")
        print(f"min_f1:{min(f1_score_list)}")
        return avg_acc,avg_f1

if __name__ == '__main__':
    trainX = ["../stock/day_train.txt", "../stock/month_train.txt", "../stock/quarter_train.txt"]
    trainY = ["../stock/day_train_la.txt", "../stock/month_train_la.txt", "../stock/quarter_train_la.txt"]
    testX = ["../stock/day_test.txt", "../stock/month_test.txt", "../stock/quarter_test.txt"]
    testY = ["../stock/day_test_la.txt", "../stock/month_test_la.txt", "../stock/quarter_test_la.txt"]

    H = 3
    nclass = 2
    iternum = 30
    dicnum = 18
    ratio = {0: 1, 1: 21, 2: 63}
    model = True_MvDPL(iternum, H, nclass, dicnum, trainX, trainY, testX, testY, ratio)
    acc, f1 = model.run()
