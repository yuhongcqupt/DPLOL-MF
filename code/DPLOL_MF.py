import numpy as np
import pandas as pd
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn import preprocessing
from functools import reduce
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import time

#### Discriminative Dictionary Pair Learning constrained by
#### Ordinal locality for Mixed Sampling Data based classification
class DPLOL_MF:

    def __init__(self,maxIter,H,nclass,dict_size,trainX,trainY,testX,testY,ratio):
        self.maxIter = maxIter  # The maximum number of iterations
        self.H = H  # 视图数
        self.nclass = nclass  # 类簇数
        self.dict_size = dict_size  # 初始化字典的原子数
        # 参数
        # tau = 0.05
        self.alpha = 0.1
        self.beta = 0.01
        self.miu = 0.001
        self.lambda1 = 0.0001  # 一个非常小的常量，保证算法的安全性 lambda1 * I_mat
        # 文件路径参数
        self.datafile = trainX
        self.labelfile = trainY
        self.testfile = testX
        self.true_testlabel_file = testY
        #视图数据比例
        self.samp_freq=ratio


    def loaddata(self,datafile):
        return np.array(pd.read_csv(datafile, sep="\t", header=None)).astype(np.float)


    # 数据处理函数（列和为1）
    def normcol_equal(self,matin):
        # matout = argmin||matout-matin||_F^2, s.t. matout(:,i)=1
        matout = matin / np.tile(np.sqrt(np.sum(np.multiply(matin, matin), axis=0) + np.spacing(1)), [np.shape(matin)[0], 1])
        return matout


# 处理列和小于1的函数
    def normcol_lessequal(self,matin):
        part = np.sqrt(np.sum(np.multiply(matin, matin), axis=0) + np.spacing(1))
        part[part < 1] = 1
        matout = matin / np.tile(part, (np.shape(matin)[0], 1))
        return matout


    def other(self,id, c):
        other_id = []
        for c1 in range(self.nclass):
            if c1 != c:
                other_id.extend(id[c])
        return other_id


    # 按列拼接
    def contacth(self,a, b):
        return np.hstack((a, b))


    # 按行拼接
    def contactv(self,a, b):
        return np.vstack((a, b))


    # 切割编码系数矩阵的列号
    def cut_X(self,class_count):
        cut_list = [0]
        for value in class_count.values():
            cut_list.append(cut_list[-1] + value)
        return cut_list


    # 行列向量相加，广播
    def bsxfun_plus(self,a):
        l = len(a)
        return np.tile(a, (1, l)) + np.tile(a.T, (l, 1))


    # 行列向量相减，广播
    def bsxfun_minus(self,a, b):
        l = np.shape(a)[1]
        return a - np.tile(b.reshape((-1, 1)), (1, l))


    # 行列向量相除，广播
    def bsxfun_divide(self,a, b):
        l = np.shape(a)[1]
        return a / np.tile(b.reshape((-1, 1)), (1, l))


    # 欧氏距离计算
    def EuDist2(self,a):
        aa = np.sum(a * a, axis=1).reshape((-1, 1))  # n*1
        ab = np.dot(a, a.T)  # n*n
        d = self.bsxfun_plus(aa) - 2 * ab
        d[d < 0] = 0
        d = np.sqrt(d)
        return np.maximum(d, d.T)


    # 向量映射，对应位置赋值1
    def assignment(self,a, b, c):
        l = len(b)
        for i in range(l):
            a[b[i], c[i]] = 1
        return a


    # 求拉普拉斯矩阵
    def tripletlap(self,D, P, k):
        # 以前k个列向量计算拉普拉斯矩阵L
        ndata = np.shape(D)[1]
        if k <= 0 or k >= ndata:
            print("parameter k is error!")

        # distance and weighting
        dist1 = self.EuDist2(D.T)  # 综合字典每一列满足局部序结构
        dist2 = self.EuDist2(P)  # 解析字典每一行满足局部序结构
        idx1 = np.argsort(dist1)
        idx1 = idx1[:, 1:k + 1]  # default: not self-connected

        idx2 = np.argsort(dist2)
        idx2 = idx2[:, 1:k + 1]  # default: not self-connected

        # G的第i行：在其他ndata-1个样本中，属于第i个样本的k个最近邻居=1；不属于=0；
        G1 = np.zeros((ndata, ndata))
        a = list(range(ndata))
        for i in range(k):
            self.assignment(G1, a, idx1[:, i])

        G2 = np.zeros((ndata, ndata))
        for i in range(k):
            self.assignment(G2, a, idx2[:, i])

        C1 = dist1 * G1
        C1 = -np.mean(np.sum(G1, axis=1)) * self.bsxfun_minus(C1, np.sum(C1, axis=1) / np.sum(G1, axis=1))
        C1 = C1 * G1

        C2 = dist2 * G2
        C2 = -np.mean(np.sum(G2, axis=1)) * self.bsxfun_minus(C2, np.sum(C2, axis=1) / np.sum(G2, axis=1))
        C2 = C2 * G2

        G = G1 + G2
        # G=G2
        G[G > 1] = 1
        C = C1 + C2
        # C=C2

        # # the following two lines aim to scale non-zero weights to [0,1]
        C = self.bsxfun_divide(self.bsxfun_minus(C, np.min(C, axis=1)), np.max(C, axis=1) - np.min(C, axis=1))
        C = C * G

        # print(C)

        # obtain Laplacian matrix
        Csym = 0.5 * (C + C.T)
        d = np.sum(Csym, axis=1)
        if np.min(np.diag(d)) > 0:
            L = np.dot(np.diag(1 / np.sqrt(d)), (np.diag(d) - Csym)), np.diag(1 / np.sqrt(d))
        else:
            L = np.diag(d) - Csym
        return np.maximum(L, L.T)


    def initilization(self, MSData, id, H, dict_size):
        # 对不同视图的数据，初始化D,P,L
        # dim = np.shape(MSData[0])[1]  # 样本的特征维度
        # I_mat1 = np.eye(dim)  # 单位矩阵
        classA = {}#样本
        classD = {}
        classP = {}
        dataInvmat = {}
        viewL = {}
        contact_D = {}
        contact_P = {}
        contact_A = {}
        for h in range(H):
            classA[h] = {}
            classD[h] = {}
            classP[h] = {}
            dataInvmat[h] = {}

            dim = np.shape(MSData[h])[1]  # 样本的特征维度
            I_mat1 = np.eye(dim)  # 单位矩阵

            for c in range(self.nclass):
                temData = MSData[h].T[:, id[h][c]]
                classA[h][c] = temData #取第c类样本
                classD[h][c] = self.normcol_equal(np.random.randn(dim, dict_size))  # 47 * 30
                classP[h][c] = self.normcol_equal(np.random.randn(dim, dict_size)).T

                # 其他类簇数据矩阵
                id_ = self.other(id[h], c)#其他类标签的列表
                # print(id_)
                temDataC = MSData[h].T[:, id_]#其他类的样本
                dataInvmat[h][c] = np.linalg.inv(
                    self.alpha * np.dot(temData, temData.T) + self.beta * np.dot(temDataC, temDataC.T) + self.lambda1 * I_mat1)
            contact_D[h] = reduce(self.contacth, [classD[h][c] for c in range(self.nclass)])#初始化结构化字典，
            contact_P[h] = reduce(self.contactv, [classP[h][c] for c in range(self.nclass)])#初始化解析字典
            contact_A[h] = reduce(self.contacth, [classA[h][c] for c in range(self.nclass)])#样本矩阵。
            viewL[h] = self.tripletlap(contact_D[h], contact_P[h], dict_size)#求拉普拉斯矩阵
        return classA, classD, classP, dataInvmat, viewL, contact_D, contact_P, contact_A


    def update_X(self,D, datamat, P, L):
        I_mat = np.eye(self.dict_size * self.nclass)
        # print(np.dot(D.T, D))
        # print( alpha * I_mat)
        # print(miu*L)
        inv_mat = np.linalg.inv((np.dot(D.T, D) + self.alpha * I_mat + self.miu * L))
        X = np.dot(inv_mat, (np.dot(D.T, datamat) + self.alpha * np.dot(P, datamat)))
        return X


    def update_P(self,X, dataInv, datamat):
        P = np.dot(self.alpha * np.dot(X, datamat.T), dataInv)
        # print(np.shape(X))
        return P


    def update_D(self,X, datamat, D):
        # 类簇系数矩阵X[1]的行
        I_mat = np.eye(np.shape(X)[0])
        rho = 1
        rate_rho = 1.2
        tempS = D
        tempT = np.zeros(np.shape(tempS))
        previousD = D
        iter = 1
        error = 1
        while error > 1e-8 and iter < 30:
            # inv = np.linalg.inv((beta * hk + rho) * I_mat + np.dot(tempX, tempX.T))
            inv = np.linalg.inv(rho * I_mat + np.dot(X, X.T))
            # print(np.shape(np.dot(tempData, tempX.T)))
            # print(np.shape(rho * (tempS - tempT)))
            part2 = np.dot(datamat, X.T) + rho * (tempS - tempT)
            tempD = np.dot(part2, inv)
            tempS = self.normcol_lessequal(tempD + tempT)
            tempT = tempT + tempD - tempS
            rho = rate_rho * rho
            # print(type(previousD - tempD))
            error = np.mean(np.mean(np.array(previousD - tempD) ** 2, axis=0))
            previousD = tempD
            iter += 1
        D = tempD
        return D


    def MSDPL(self,MSData, id, H, class_count):
        # 初始化综合字典 D和分析字典 P 同时利用更新公式得到系数矩阵 X
        A, D, P, dataInv, viewL, contact_D, contact_P, contact_A = self.initilization(MSData, id, H, self.dict_size)
        ms_iter = 1
        X = {}#编码系数矩阵
        classX = {}
        cut_list = {}
        for h in range(H):
            cut_list[h] = self.cut_X(class_count[h])
        while ms_iter < self.maxIter:
            for h in range(H):
                X[h] = self.update_X(contact_D[h], contact_A[h], contact_P[h], viewL[h])  # 更新系数矩阵
                classX[h] = {}
                for c in range(self.nclass):
                    classX[h][c] = X[h][c * self.dict_size:(c + 1) * self.dict_size:, cut_list[h][c]:cut_list[h][c + 1]]  # 切割方法

                    P[h][c] = self.update_P(classX[h][c], dataInv[h][c], A[h][c])  # 更新分析字典矩阵
                    D[h][c] = self.update_D(classX[h][c], A[h][c], D[h][c])  # 更新综合字典矩阵
                # 拼接 D P
                contact_D[h] = reduce(self.contacth, [D[h][c] for c in range(self.nclass)])
                contact_P[h] = reduce(self.contactv, [P[h][c] for c in range(self.nclass)])
                viewL[h] = self.tripletlap(contact_D[h], contact_P[h], self.dict_size)
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


    def classification_ADL(self,test_data, D, P, samp_freq):
        sample_num = np.shape(test_data[0])[1]
        a = list(range(sample_num))
        f = {}
        for h in range(self.H):
            sep = int(samp_freq[h] * 100)
            b = []
            for i in range(sample_num // 100):
                b.extend(a[i * 100:i * 100 + sep])
                f[h] = b

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
            for c in range(self.nclass):
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
        MSData = {}#训练集
        z = {}#测试集
        label = {}#训练集标签
        h_dimension = {}#训练集维度
        true_testlabel = {}#测试集标签
        for h in range(self.H):
            MSData[h] = self.loaddata(self.datafile[h])
            z[h] = self.loaddata(self.testfile[h])
            h_dimension[h] = np.shape(MSData[h])[1]
            label[h] = list(np.array(pd.read_csv(self.labelfile[h], sep="\t", header=None)).astype(np.int)[0, :])
            true_testlabel[h] = list(
                np.array(pd.read_csv(self.true_testlabel_file[h], sep="\t", header=None)).astype(np.int)[0, :])

        # 转换成非负 归一化0-1 降维
        scaler = MinMaxScaler()

        MSData, label = self.sample_alg(MSData, label, self.samp_freq)
        z, true_testlabel = self.sample_alg(z, true_testlabel, self.samp_freq)

        for h in range(self.H):
            scaler.fit(MSData[h])
            MSData[h] = scaler.transform(MSData[h])
            scaler.fit(z[h])
            z[h] = scaler.transform(z[h]).T


        # 为计算其他类簇矩阵作准备
        id = {}
        class_count = {}  # class_view_count={}
        for h in range(self.H):
            id[h] = {}
            class_count[h] = {}
            for c in range(self.nclass):
                id[h][c] = [i for i, x in enumerate(label[h]) if x == c]#三维的数组。视图h中c类样本是哪些。
                class_count[h][c] = len(id[h][c])#视图h中c类样本的个数
        acc_list = []
        f1_score_list = []
        for i in range(10):
            # 字典学习阶段
            tr_start = time.clock()
            D, P = self.MSDPL(MSData, id, self.H, class_count)
            tr_time = (time.clock() - tr_start)
            # print(D, P)

            # 测试样本打标
            tt_start = time.clock()
            test_label = self.classification_ADL(z, D, P, self.samp_freq)
            tt_time = (time.clock() - tt_start)

            acc_list.append(accuracy_score(true_testlabel[0], test_label))
            f1_score_list.append(metrics.f1_score(true_testlabel[0], test_label, average='weighted'))

            print("acc:", acc_list[i])
            print("f1_score:", f1_score_list[i])

            print("train time used:", tr_time)
            print("test time used:", tt_time)
        avg_acc = np.average(acc_list)
        avg_f1_score = np.average(f1_score_list)
        std_acc = np.std(acc_list)
        std_f1 = np.std(f1_score_list)
        print(f"std_acc:{std_acc}")
        print(f"std_f1:{std_f1}")
        print(f"max_acc:{max(acc_list)}")
        print(f"max_f1:{max(f1_score_list)}")
        print(f"avarge_acc:{avg_acc}")
        print(f"avarge_f1:{avg_f1_score}")
        print(f"min_acc:{min(acc_list)}")
        print(f"min_f1:{min(f1_score_list)}")
        return avg_acc, avg_f1_score

if __name__ == '__main__':

     # trainX = [".\Caltech101-7\\tr_Gabor.txt", ".\Caltech101-7\\tr_WM.txt"]
     # trainY = [".\Caltech101-7\\trl_7-Label.txt", ".\Caltech101-7\\trl_7-Label.txt"]
     # testX = [".\Caltech101-7\\t_Gabor.txt", ".\Caltech101-7\\t_WM.txt"]
     # testY = [".\Caltech101-7\\tl_7-Label.txt", ".\Caltech101-7\\tl_7-Label.txt"]

     # trainX = [".\digit\\tr_fou.txt", ".\digit\\tr_kar.txt", ".\digit\\tr_zer.txt"]
     # trainY = [".\digit\\trl_3view.txt", ".\digit\\trl_3view.txt", ".\digit\\trl_3view.txt"]
     # testX = [".\digit\\t_fou.txt", ".\digit\\t_kar.txt", ".\digit\\t_zer.txt"]
     # testY = [".\digit\\tl_3view.txt", ".\digit\\tl_3view.txt", ".\digit\\tl_3view.txt"]

     # trainX = [".\\Caltech101-7\\tr_Gabor.txt", ".\\Caltech101-7\\tr_WM.txt"]
     # trainY = [".\\Caltech101-7\\trl_7-Label.txt", ".\\Caltech101-7\\trl_7-Label.txt"]
     # testX = [".\\Caltech101-7\\t_Gabor.txt", ".\\Caltech101-7\\t_WM.txt"]
     # testY = [".\\Caltech101-7\\tl_7-Label.txt", ".\\Caltech101-7\\tl_7-Label.txt"]

     # trainX = [".\Caltech101-20\\tr_Gabor.txt", ".\Caltech101-20\\tr_WM.txt"]
     # trainY = [".\Caltech101-20\\trl_20-Label.txt", ".\Caltech101-20\\trl_20-Label.txt"]
     # testX = [".\Caltech101-20\\t_Gabor.txt", ".\Caltech101-20\\t_WM.txt"]
     # testY = [".\Caltech101-20\\tl_20-Label.txt", ".\Caltech101-20\\tl_20-Label.txt"]

     trainX = [".\SensIT\\tr_first.txt", ".\SensIT\\tr_second.txt"]
     trainY = [".\SensIT\\trl_label.txt", ".\SensIT\\trl_label.txt"]
     testX = [".\SensIT\\t_first.txt", ".\SensIT\\t_second.txt"]
     testY = [".\SensIT\\tl_label.txt", ".\SensIT\\tl_label.txt"]

     H = 2
     nclass = 3
     iternum = 30
     dicnum = 18

     ratio={0:1,1:0.4,2:1}
     model=DPLOL_MF(iternum, H, nclass, dicnum, trainX, trainY, testX, testY,ratio)
     avg_acc, avg_f1_score = model.run()


