import numpy as np,math
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib import pyplot as pt


def dataPreprocessing(i):
    for j in range(3, 14):
        temp = 3
        add = 0
        for k in range(1, 4):
            if (math.isnan(data[i - k][j])):
                temp = temp - 1
            else:
                add = add + data[i - k][j]
        for k in range(1, 4):
            if (math.isnan(data[i - k][j]) and temp == 0):
                try:
                    data[i - k][j] = 0
                except:
                    print("err")
            elif (math.isnan(data[i - k][j])):
                try:
                    data[i - k][j] = round(add / temp)
                except:
                    print("err")

def cholestrolProcessing(i):
    if (math.isnan(data[i - 1][14])== False):
        data[i - 1][16]=round((data[i - 1][13]-(data[i - 1][14]+data[i - 1][15]))*5)

        for j in range (2,4):
            for k in range(14,17):
                data[i - j][k] =round((data[i - j][13]*data[i - 1][k])/data[i - 1][13])


        p3.append(data[i - 3][1:22])
        p3.append(data[i - 2][1:22])
        p3.append(data[i - 1][1:22])

def dataScoring(out):
    sum = 0
    for i in range(0, len(p3)):
        if (p3[i][out] == 1):
            sum += 1
        for j in range(0, 16):
            for k in range(0, len(feature[j])):
                if (p3[i][j] <= feature[j][k] and p3[i][out] == 1 and k < len(feature[j]) - 1):
                    featureScore[j][k] = featureScore[j][k] + 1
                    break
                if (p3[i][j] >= feature[j][k] and p3[i][out] == 1 and k == len(feature[j]) - 1):
                    featureScore[j][k] = featureScore[j][k] + 1
                    break
                if (i == len(p3) - 1):
                    featureScore[j][k]=round(featureScore[j][k] / sum,3)
                    tempFeature.append(featureScore[j][k])

def setScorring():
    for i in range(0, len(p3)):
        for j in range(0, 16):
            for k in range(0, len(feature[j])):
                if (p3[i][j] <= feature[j][k] and  k < len(feature[j]) - 1):
                    p3[i][j]= featureScore[j][k]
                    break
                if (p3[i][j] >= feature[j][k] and k == len(feature[j]) - 1):
                    p3[i][j]= featureScore[j][k]
                    break
        if((i+1)%3==0):
            temp = []
            temp.extend(p3[i - 2][0:16])
            temp.extend(p3[i - 1][0:16])
            temp.extend(p3[i ][0:21])
            finalData.append(temp)
allFeatureScore=[]
score=[]
for j in range(0, 4):
    data = np.genfromtxt('Data/data.csv', delimiter=',')
    count = 0
    cmp = data[0][0]
    cycle = 0
    p1Count = 0
    p2Count = 0
    p3Count = 0
    p3 = []
    p2 = []
    p1 = []
    finalData=[]
    tempFeature=[]
    feature = [[1, 2], [30, 40, 50, 60, 60], [16, 18, 20, 23, 26, 26], [120, 130, 140, 150, 150],
               [80, 90, 120, 120], [80, 115, 130, 150, 150]
        , [0, 1], [100, 125, 125], [0, 1], [1, 2, 3, 4], [0, 1], [0, 2, 10, 15, 15], [200, 240, 240],
               [100, 130, 160, 190, 190], [45, 55, 55], [150, 200, 250, 250]]

    featureScore = [[0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]
        , [0, 0], [0, 0, 0], [0, 0], [0, 0, 0, 0], [0, 0], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0],
                    [0, 0, 0, 0]]
    for i in range(0, len(data)):
        if cmp != data[i][0]:
            # print(cycle)
            if cycle == 3:
                dataPreprocessing(i)
                cholestrolProcessing(i)
                p3Count += 1
            elif cycle == 2:
                p2.append(data[i - 2])
                p2.append(data[i - 1])
                p2Count += 1
            elif cycle == 1:
                p1.append(data[i - 1])
                p1Count += 1
            cmp = data[i][0]
            count += 1
            cycle = 1
        else:
            cycle += 1
    np.savetxt("Data/p1.csv", p1, delimiter=',')
    np.savetxt("Data/p2.csv", p2, delimiter=',')
    np.savetxt("Data/p3.csv", p3, delimiter=',')


    # scaler = MinMaxScaler()
    # p3 = scaler.fit_transform(p3)
    # count=0
    dataFile=["Data/cvd_data.csv","Data/str_data.csv","Data/dr_data.csv","Data/fcd_data.csv"]
    dataScoring(16+j)
    setScorring()
    print(featureScore)
    allFeatureScore.append(tempFeature)
    count1=count2=0
    sum1=sum2=0
    for i in range(0,len(finalData)):
        #print(repr(round(sum(finalData[i][0:48]))) + " --- " + repr(finalData[i][48 + j]))
        if(sum(finalData[i][0:48])>=23.9):
            finalData[i][48 + j]=1
            sum1 += (sum(finalData[i][0:48]))
            count1 += 1
        else:
            finalData[i][48 + j] = 0
            sum2 += (sum(finalData[i][0:48]))
            count2 += 1
    score.append([1,count1,(sum1/count1),0, count2, (sum2 / count2)])
    print(repr(count1)+"- 1 - Score: "+ repr((sum1/count1)))
    print(repr(count2) + "- 0 - Score: " + repr((sum2 / count2)))

    np.savetxt(dataFile[j], finalData, delimiter=',')

print(allFeatureScore)
np.savetxt("Data/All_Feature_Score.csv", allFeatureScore, delimiter=',')
np.savetxt("Data/Score.csv", score, delimiter=',')