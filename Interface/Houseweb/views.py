from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json
import model.test as mltest
import model.utils as mdul
from model.floorplan import *
import retrieval.retrieval as rt
import time
import pickle
import scipy.io as sio
import numpy as np
from model.decorate import *
import math
import pandas as pd
import matlab.engine
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from g2p.plotcopy import plot_fp
import base64
from io import BytesIO
import plotly.graph_objects as go

global test_data, test_data_topk, testNameList, trainNameList
global train_data,  trainTF, train_data_eNum, train_data_rNum
global engview, model
global tf_train, centroids, clusters


def home(request):
    return render(request, "home.html", )


def Init(request):
    start = time.perf_counter()
    getTestData()
    getTrainData()
    loadMatlabEng()
    loadModel()
    loadRetrieval()
    end = end = time.perf_counter()

    print('Init(model+test+train+engine+retrieval) time: %s Seconds' % (end - start))

    return HttpResponse(None)


def loadMatlabEng():
    startengview = time.perf_counter()

    global engview
    engview = matlab.engine.start_matlab()
    engview.addpath(r'./align_fp/', nargout=0)
    endengview = time.perf_counter()
    print(' matlab.engineview time: %s Seconds' % (endengview - startengview))


def loadRetrieval():
    global tf_train, centroids, clusters
    t1 = time.perf_counter()
    tf_train = np.load(r'D:\MajorProject\Florex\Interface\retrieval\tf_train.npy')
    centroids = np.load(r'D:\MajorProject\Florex\Interface\retrieval\centroids_train.npy')
    clusters = np.load(r'D:\MajorProject\Florex\Interface\retrieval\clusters_train.npy')
    t2 = time.perf_counter()
    print('load tf/centroids/clusters', t2 - t1)


def getTestData():
    start = time.perf_counter()

    global test_data, testNameList, trainNameList
 
    test_data = pickle.load(open('./static/Data/data_test_converted.pkl', 'rb'))
    test_data, testNameList, trainNameList = test_data['data'], list(test_data['testNameList']), list(
        test_data['trainNameList'])
    end = time.perf_counter()
    print('getTestData time: %s Seconds' % (end - start))


def getTrainData():
    start = time.perf_counter()
    global train_data, trainNameList, trainTF, train_data_eNum, train_data_rNum
    
    train_data = pickle.load(open('./static/Data/data_train_converted.pkl', 'rb'))
    train_data, trainNameList, trainTF = train_data['data'], list(train_data['nameList']), list(train_data['trainTF'])
    
    train_data_eNum = pickle.load(open('./static/Data/data_train_eNum.pkl', 'rb'))
    train_data_eNum = train_data_eNum['eNum']
    train_data_rNum = np.load('./static/Data/rNum_train.npy')
    end = time.perf_counter()
    print('getTrainData time: %s Seconds' % (end - start))


def loadModel():
    global model, train_data, trainNameList
    start = time.perf_counter()
    model = mltest.load_model()
    end = time.perf_counter()
    print('loadModel time: %s Seconds' % (end - start))
    start = time.perf_counter()
    test = train_data[trainNameList.index("75119")]
    mltest.test(model, FloorPlan(test, train=True))
    end = time.perf_counter()
    print('test Model time: %s Seconds' % (end - start))


def LoadTestBoundary(request):
    start = time.perf_counter()

    testName = request.GET.get('testName').split(".")[0]
    test_index = testNameList.index(testName)
    data = test_data[test_index]
    data_js = {}
    data_js["door"] = str(data.boundary[0][0]) + "," + str(data.boundary[0][1]) + "," + str(
        data.boundary[1][0]) + "," + str(data.boundary[1][1])
    ex = ""
    for i in range(len(data.boundary)):
        ex = ex + str(data.boundary[i][0]) + "," + str(data.boundary[i][1]) + " "
    data_js['exterior'] = ex
    end = time.perf_counter()
    print('LoadTestBoundary time: %s Seconds' % (end - start))
    return HttpResponse(json.dumps(data_js), content_type="application/json")


def get_filter_func(mask, acc, num):
    filters = [
        None if not mask else (
            np.equal if acc[i] else np.greater_equal
        )
        for i in range(len(mask))
    ]

    def filter_func(data):
        for i in range(len(filters)):
            if (filters[i] is not None) and (not filters[i](data[i], num[i])): return False
        return True

    return filter_func


def filter_graph(graph_):
    filters = graph_

    def filter_graphfunc(data):
        sub = data - filters
        return ((sub >= 0).all())

    return filter_graphfunc


def NumSearch(request):
    start = time.perf_counter()

    data_new = json.loads(request.GET.get("userInfo"))
    testName = data_new[0].split(".")[0]
    test_index = testNameList.index(testName)
    topkList = []
    topkList.clear()
    data = test_data[test_index]

   
    multi_clusters=False
    test_data_topk = rt.retrieval(data, 1000,multi_clusters)
    
    if len(data_new) > 1:
        roomactarr = data_new[1]
        roomexaarr = data_new[2]
        roomnumarr = [int(x) for x in data_new[3]]
        
        test_num = train_data_rNum[test_data_topk]
        filter_func = get_filter_func(roomactarr, roomexaarr, roomnumarr)
        indices = np.where(list(map(filter_func, test_num)))
        indices = list(indices)
        if len(indices[0]) < 20:
            topk = len(indices[0])
        else:
            topk = 20
        topkList.clear()
        for i in range(topk):
            topkList.append(str(trainNameList[int(test_data_topk[indices[0][i]])]) + ".png")
    end = time.perf_counter()
    print('NumberSearch time: %s Seconds' % (end - start))
    return HttpResponse(json.dumps(topkList), content_type="application/json")


def FindTraindata(trainname):
    start = time.perf_counter()

    train_index = trainNameList.index(trainname)
    data = train_data[train_index]
    data_js = {}
    data_js["hsname"] = trainname

    data_js["door"] = str(data.boundary[0][0]) + "," + str(data.boundary[0][1]) + "," + str(
        data.boundary[1][0]) + "," + str(data.boundary[1][1])
    print("testboundary", data_js["door"])
    ex = ""
    for i in range(len(data.boundary)):
        ex = ex + str(data.boundary[i][0]) + "," + str(data.boundary[i][1]) + " "
    data_js['exterior'] = ex

    data_js["hsedge"] = [[int(u), int(v)] for u, v in data.edge[:, [0, 1]]]

    hsbox = [[[float(x1), float(y1), float(x2), float(y2)], [mdul.room_label[cate][1]]] for
             x1, y1, x2, y2, cate in data.box[:]]
    external = np.asarray(data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    
    area_ = (ymax - ymin) * (xmax - xmin)
    
    data_js["rmsize"] = [
        [[20 * math.sqrt((float(x2) - float(x1)) * (float(y2) - float(y1)) / float(area_))], [mdul.room_label[cate][1]]]
        for
        x1, y1, x2, y2, cate in data.box[:]]
   

    box_order = data.order
    data_js["hsbox"] = []
    for i in range(len(box_order)):
        data_js["hsbox"].append(hsbox[int(float(box_order[i])) - 1])

    data_js["rmpos"] = [[int(cate), str(mdul.room_label[cate][1]), float((x1 + x2) / 2), float((y1 + y2) / 2)] for
                        x1, y1, x2, y2, cate in data.box[:]]
    end = time.perf_counter()
    print('find train data time: %s Seconds' % (end - start))
    return data_js


def LoadTrainHouse(request):
    trainname = request.GET.get("roomID").split(".")[0]
    data_js = FindTraindata(trainname)
    return HttpResponse(json.dumps(data_js), content_type="application/json")


'''
 transfer the graph of the training data into the graph of the test data
'''


def TransGraph(request):
    start = time.perf_counter()

    userInfo = request.GET.get("userInfo")
    testname = userInfo.split(',')[0]
    trainname = request.GET.get("roomID")
    mlresult = mltest.get_userinfo(testname, trainname)

    fp_end = mlresult
   
    sio.savemat("./static/" + userInfo.split(',')[0].split('.')[0] + ".mat", {"data": fp_end.data})

    data_js = {}
    # fp_end  hsedge
    data_js["hsedge"] = (fp_end.get_triples(tensor=False)[:, [0, 2, 1]]).astype(float).tolist()

    # fp_rmsize
    external = np.asarray(fp_end.data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    area_ = (ymax - ymin) * (xmax - xmin)
    data_js["rmsize"] = [
        [[20 * math.sqrt((float(x2) - float(x1)) * (float(y2) - float(y1)) / float(area_))], [mdul.room_label[cate][1]]]
        for
        x1, y1, x2, y2, cate in fp_end.data.box[:]]
    # fp_end rmpos

    rooms = fp_end.get_rooms(tensor=False)

    
    center = [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in fp_end.data.box[:, :4]]

    # boxes_pred
    data_js["rmpos"] = []
    for k in range(len(center)):
        node = float(rooms[k]), mdul.room_label[int(rooms[k])][1], center[k][0], center[k][1], float(k)
        data_js["rmpos"].append(node)

    test_index = testNameList.index(testname.split(".")[0])
    data = test_data[test_index]
    ex = ""
    for i in range(len(data.boundary)):
        ex = ex + str(data.boundary[i][0]) + "," + str(data.boundary[i][1]) + " "
    data_js['exterior'] = ex
    data_js["door"] = str(data.boundary[0][0]) + "," + str(data.boundary[0][1]) + "," + str(
        data.boundary[1][0]) + "," + str(data.boundary[1][1])
    end = time.perf_counter()
    print('TransGraph time: %s Seconds' % (end - start))
    return HttpResponse(json.dumps(data_js), content_type="application/json")


def AdjustGraph(request):
    start = time.perf_counter()

    # newNode index-typename-cx-cy
    # oldNode index-typename-cx-cy
    # newEdge u-v
    NewGraph = json.loads(request.GET.get("NewGraph"))
    testname = request.GET.get("userRoomID")
    trainname = request.GET.get("adptRoomID")
    s = time.perf_counter()
    mlresult = mltest.get_userinfo_adjust(testname, trainname, NewGraph)
    e = time.perf_counter()
    print('get_userinfo_adjust: %s Seconds' % (e - s))
    fp_end = mlresult[0]
    global boxes_pred
    boxes_pred = mlresult[1]
    
    data_js = {}
    data_js["hsedge"] = (fp_end.get_triples(tensor=False)[:, [0, 2, 1]]).astype(float).tolist()
  
    rooms = fp_end.get_rooms(tensor=False)
    center = [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in fp_end.data.box[:, :4]]

    box_order = mlresult[2]
    '''
    handle the information of the room boxes 
    boxes_pred: the prediction from net
    box_order: The order in which boxes are drawn

    '''
    room = []
    for o in range(len(box_order)):
        room.append(float((rooms[int(float(box_order[o][0])) - 1])))
    boxes_end = []
    for i in range(len(box_order)):
        tmp = []
        for j in range(4):
            tmp.append(float(boxes_pred[int(float(box_order[i][0])) - 1][j]))
        boxes_end.append(tmp)
    
    data_js['roomret'] = []
    for k in range(len(room)):
        data = boxes_end[k], [mdul.room_label[int(room[k])][1]], box_order[k][0] - 1
        data_js['roomret'].append(data)
    
    # change the box size
    global relbox
    relbox = data_js['roomret']
    global reledge
    reledge = data_js["hsedge"]

    test_index = testNameList.index(testname.split(".")[0])
    data = test_data[test_index]
    ex = ""
    for i in range(len(data.boundary)):
        ex = ex + str(data.boundary[i][0]) + "," + str(data.boundary[i][1]) + " "
    data_js['exterior'] = ex
    data_js["door"] = str(data.boundary[0][0]) + "," + str(data.boundary[0][1]) + "," + str(
        data.boundary[1][0]) + "," + str(data.boundary[1][1])

    external = np.asarray(data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    area_ = (ymax - ymin) * (xmax - xmin)
    data_js['rmsize'] = []
    for i in range(len(data_js['roomret'])):
        rmsize = 20 * math.sqrt((float(data_js['roomret'][i][0][2]) - float(data_js['roomret'][i][0][0])) * (
                float(data_js['roomret'][i][0][3]) - float(data_js['roomret'][i][0][1])) / float(area_)), \
                 data_js["roomret"][i][1][0]
        data_js["rmsize"].append(rmsize)

    data_js["rmpos"] = []

    newGraph = NewGraph[0]
    for i in range(len(data_js['roomret'])):
        for k in range(len(newGraph)):
            if (data_js['roomret'][i][1][0] == newGraph[k][1]):
                x_center = int((data_js['roomret'][i][0][0] + data_js['roomret'][i][0][2]) / 2)
                y_center = int((data_js['roomret'][i][0][1] + data_js['roomret'][i][0][3]) / 2)
                x_graph = newGraph[k][2]
                y_graph = newGraph[k][3]
                if ((int(x_graph - 30) < x_center < int(x_graph + 30))):
                    node = float(rooms[k]), newGraph[k][1], x_center, y_center, float(
                        newGraph[k][0])
                    data_js["rmpos"].append(node)
                    newGraph.pop(k)
                    break
                if ((int(y_graph - 30) < y_center < int(y_graph + 30))):
                    node = float(rooms[k]), newGraph[k][1], x_center, y_center, float(
                        newGraph[k][0])
                    data_js["rmpos"].append(node)
                    newGraph.pop(k)

                    break
    
    fp_end.data = add_dw_fp(fp_end.data)
    data_js["indoor"] = []
    
    boundary = data.boundary
    
    isNew = boundary[:, 3]
    frontDoor = boundary[[0, 1]]  
    frontDoor = frontDoor[:, [0, 1]]  
    frontsum = frontDoor.sum(axis=1).tolist()
    idx = frontsum.index(min(frontsum))
    wallThickness = 3
    if idx == 1:
        frontDoor = frontDoor[[1, 0], :]
    orient = boundary[0][2]
    if orient == 0 or orient == 2:
        frontDoor[0][0] = frontDoor[0][0] + wallThickness / 4
        frontDoor[1][0] = frontDoor[1][0] - wallThickness / 4
    if orient == 1 or orient == 3:
        frontDoor[0][1] = frontDoor[0][1] + wallThickness / 4
        frontDoor[1][1] = frontDoor[1][1] - wallThickness / 4
    

    data_js["windows"] = []
    for indx, x, y, w, h, r in fp_end.data.windows:
        if w != 0:
            tmp = [x + 2, y - 2, w - 2, 4]
            data_js["windows"].append(tmp)
        if h != 0:
            tmp = [x - 2, y, 4, h]
            data_js["windows"].append(tmp)
    data_js["windowsline"] = []
    for indx, x, y, w, h, r in fp_end.data.windows:
        if w != 0:
            tmp = [x + 2, y, w + x, y]
            data_js["windowsline"].append(tmp)
        if h != 0:
            tmp = [x, y, x, h + y]
            data_js["windowsline"].append(tmp)
    
    sio.savemat("./static/" + testname.split(',')[0].split('.')[0] + ".mat", {"data": fp_end.data})

    end = time.perf_counter()
    print('AdjustGraph time: %s Seconds' % (end - start))
    return HttpResponse(json.dumps(data_js), content_type="application/json")


def RelBox(request):
    id = request.GET.get("selectRect")
    print(id)
    global relbox
    global reledge
    rdirgroup=get_dir(id,relbox,reledge)
    return HttpResponse(json.dumps(rdirgroup), content_type="application/json")

def get_dir(id,relbox,reledge):
    rel = []
    selectindex = int(id.split("_")[1])
    select = np.zeros(4).astype(int)
    for i in range(len(relbox)):
        a = math.ceil(relbox[i][0][0]), math.ceil(relbox[i][0][1]), math.ceil(relbox[i][0][2]), math.ceil(
            relbox[i][0][3]), int(relbox[i][2])
        rel.append(a)
        if (selectindex == int(relbox[i][2])):
            # select:x1,x0,y0,y1.relbox:x0,y0,x1,y1
            select[0] = math.ceil(relbox[i][0][2])
            select[1] = math.ceil(relbox[i][0][0])
            select[2] = math.ceil(relbox[i][0][1])
            select[3] = math.ceil(relbox[i][0][3])
    rel = np.array(rel)
    df = pd.DataFrame({'x0': rel[:, 0], 'y0': rel[:, 1], 'x1': rel[:, 2], 'y1': rel[:, 3], 'rindex': rel[:, 4]})
    group_label = [(0, 'x1', "right"),
                   (1, 'x0', "left"),
                   (2, 'y0', "top"),
                   (3, 'y1', "down")]
    dfgroup = []
    for i in range(len(group_label)):
        dfgroup.append(df.groupby(group_label[i][1], as_index=True).get_group(name=select[i]))
    rdirgroup = []
    for i in range(len(dfgroup)):
        dir = dfgroup[i]
        rdir = []
        for k in range(len(dir)):
            idx = (dir.loc[dir['rindex'] == (dir.iloc[[k]].values)[0][4]].index.values)[0]
            rdir.append(relbox[idx][1][0].__str__() + "_" + (dir.iloc[[k]].values)[0][4].__str__())
        rdirgroup.append(rdir)
    reledge = np.array(reledge)
    data1 = reledge[np.where((reledge[:, [0]] == selectindex))[0]]
    data2 = reledge[np.where((reledge[:, [1]] == selectindex))[0]]
    reledge1 = np.vstack((data1, data2))
    return rdirgroup
def Save_Editbox(request):
    global indxlist,boxes_pred
    NewGraph = json.loads(request.GET.get("NewGraph"))
    NewLay = json.loads(request.GET.get("NewLay"))
    userRoomID = request.GET.get("userRoomID")
    adptRoomID = request.GET.get("adptRoomID")
    
    NewLay=np.array(NewLay)
    NewLay=NewLay[np.argsort(NewLay[:, 1])][:,2:]
    NewLay=NewLay.astype(float).tolist()

    test_index = testNameList.index(userRoomID.split(".")[0])
    test_ = test_data[test_index]
    
    Boundary = test_.boundary
    boundary=[[float(x),float(y),float(z),float(k)] for x,y,z,k in list(Boundary)]
    test_fp =FloorPlan(test_)

    train_index = trainNameList.index(adptRoomID.split(".")[0])
    train_ =train_data[train_index]
    train_fp =FloorPlan(train_,train=True)
    fp_end = test_fp.adapt_graph(train_fp)
    fp_end.adjust_graph()
    newNode = NewGraph[0]
    newEdge = NewGraph[1]
    oldNode = NewGraph[2]
    temp = []
    for newindx, newrmname, newx, newy,scalesize in newNode:
        for type, oldrmname, oldx, oldy, oldindx in oldNode:
            if (int(newindx) == oldindx):
                tmp=int(newindx), (newx - oldx), ( newy- oldy),float(scalesize)
                temp.append(tmp)
    newbox=[]
    if mltest.adjust==True:
        oldbox = []
        for i in range(len(boxes_pred)):
            indxtmp=[boxes_pred[i][0],boxes_pred[i][1],boxes_pred[i][2],boxes_pred[i][3],boxes_pred[i][0]]
            oldbox.append(indxtmp)
    if mltest.adjust==False:
        indxlist=[]
        oldbox=fp_end.data.box.tolist()
        for i in range(len(oldbox)):
            indxlist.append([oldbox[i][4]])
        indxlist=np.array(indxlist)
        adjust=True
    oldbox=fp_end.data.box.tolist()
    X=0
    Y=0
    for i in range(len(oldbox)):
        X= X+(oldbox[i][2]-oldbox[i][0])
        Y= Y+(oldbox[i][3]-oldbox[i][1])
    x_ave=(X/len(oldbox))/2
    y_ave=(Y/len(oldbox))/2

    index_mapping = {}
    #  The room that already exists
    #  Move: Just by the distance
    for newindx, tempx, tempy,scalesize in temp:
        index_mapping[newindx] = len(newbox)
        tmpbox=[]
        scalesize = int(scalesize)
        if scalesize<1:
            scale = math.sqrt(scalesize)
            scalex = (oldbox[newindx][2] - oldbox[newindx][0]) * (1 - scale) / 2
            scaley = (oldbox[newindx][3] - oldbox[newindx][1]) * (1 - scale) / 2
            tmpbox = [(oldbox[newindx][0] + tempx) + scalex, (oldbox[newindx][1] + tempy)+scaley,
                      (oldbox[newindx][2] + tempx) - scalex, (oldbox[newindx][3] + tempy) - scaley, oldbox[newindx][4]]
        if scalesize == 1:
            tmpbox = [(oldbox[newindx][0] + tempx) , (oldbox[newindx][1] + tempy) ,(oldbox[newindx][2] + tempx), (oldbox[newindx][3] + tempy), oldbox[newindx][4]]

        if scalesize>1:
            scale=math.sqrt(scalesize)
            scalex = (oldbox[newindx][2] - oldbox[newindx][0]) * ( scale-1) / 2
            scaley = (oldbox[newindx][3] - oldbox[newindx][1]) * (scale-1) / 2
            tmpbox = [(oldbox[newindx][0] + tempx) - scalex, (oldbox[newindx][1] + tempy) - scaley,
                      (oldbox[newindx][2] + tempx) + scalex, (oldbox[newindx][3] + tempy) + scaley, oldbox[newindx][4]]

        newbox.append(tmpbox)

    #  The room just added
    #  Move: The room node with the average size of the existing room
    for newindx, newrmname, newx, newy,scalesize in newNode:
        if int(newindx)>(len(oldbox)-1):
            scalesize=int(scalesize)
            index_mapping[int(newindx)] = (len(newbox))
            tmpbox=[]
            if scalesize < 1:
                scale = math.sqrt(scalesize)
                scalex = x_ave * (1 - scale) / 2
                scaley = y_ave* (1 - scale) / 2
                tmpbox = [(newx-x_ave) +scalex,(newy-y_ave) +scaley,(newx+x_ave)-scalex,(newy+y_ave)-scaley,vocab['object_name_to_idx'][newrmname]]

            if scalesize == 1:
                tmpbox = [(newx - x_ave), (newy - y_ave), (newx + x_ave), (newy + y_ave),vocab['object_name_to_idx'][newrmname]]
            if scalesize > 1:
                scale = math.sqrt(scalesize)
                scalex = x_ave * (scale - 1) / 2
                scaley = y_ave * (scale - 1) / 2
                tmpbox = [(newx-x_ave) - scalex, (newy-y_ave)  - scaley,(newx+x_ave) + scalex, (newy+y_ave) + scaley,vocab['object_name_to_idx'][newrmname]]
            # tmpboxin = [(newx-x_ave) ,(newy-y_ave) ,(newx+x_ave) ,(newy+y_ave) ,vocab['object_name_to_idx'][newrmname]]
            # print(tmpboxin)
            # print(tmpbox)
            # print(scalesize)
            newbox.append(tmpbox)

    fp_end.data.box=np.array(newbox)
    
    adjust_Edge=[]
    for u, v in newEdge:
        tmp=[index_mapping[int(u)],index_mapping[int(v)], 0]
        adjust_Edge.append(tmp)
    fp_end.data.edge=np.array(adjust_Edge)
    rType = fp_end.get_rooms(tensor=False)

    rEdge = fp_end.get_triples(tensor=False)[:, [0, 2, 1]]
    Edge = [[float(u), float(v), float(type2)] for u, v, type2 in rEdge]
    Box=NewLay
    boundary_mat = matlab.double(boundary)
    rType_mat = matlab.double(rType.tolist())
    Edge_mat = matlab.double(Edge)
    Box_mat=matlab.double(Box)
    fp_end.data.boundary =np.array(boundary)
    fp_end.data.rType =np.array(rType).astype(int)
    fp_end.data.refineBox=np.array(Box)
    fp_end.data.rEdge=np.array(Edge)

    box_refine = engview.align_fp(boundary_mat, Box_mat,  rType_mat,Edge_mat ,18,False, nargout=3)
    box_out=box_refine[0]
    box_order=box_refine[1]
    rBoundary=box_refine[2]
    fp_end.data.newBox = np.array(box_out)
    fp_end.data.order = np.array(box_order)
    fp_end.data.rBoundary = [np.array(rb) for rb in rBoundary]
    fp_end.data = add_dw_fp(fp_end.data)
    sio.savemat("./static/" + userRoomID + ".mat", {"data": fp_end.data})
    flag=1
    return HttpResponse(json.dumps(flag), content_type="application/json")


def TransGraph_net(request):
    userInfo = request.GET.get("userInfo")
    testname = userInfo.split(',')[0]
    trainname = request.GET.get("roomID")
    mlresult = mltest.get_userinfo_net(testname, trainname)

    fp_end = mlresult[0]
    boxes_pred = mlresult[1]

    data_js = {}
    # fp_end  hsedge
    data_js["hsedge"] = (fp_end.get_triples(tensor=False)[:, [0, 2, 1]]).astype(float).tolist()

    # fp_end rmpos
    rooms = fp_end.get_rooms(tensor=False)
    room = rooms
    center = [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in fp_end.data.box[:, :4]]

    

    # boxes_pred
    data_js["rmpos"] = []
    for k in range(len(center)):
        node = float(room[k]), mdul.room_label[int(room[k])][1], center[k][0], center[k][1]
        data_js["rmpos"].append(node)
    boxes_end = boxes_pred.tolist()
    data_js['roomret'] = []
    for k in range(len(room)):
        data = boxes_end[k], [mdul.room_label[int(room[k])][1]]
        data_js['roomret'].append(data)

    test_index = testNameList.index(testname.split(".")[0])
    data = test_data[test_index]
    ex = ""
    for i in range(len(data.boundary)):
        ex = ex + str(data.boundary[i][0]) + "," + str(data.boundary[i][1]) + " "
    data_js['exterior'] = ex
    x0, x1 = np.min(data.boundary[:, 0]), np.max(data.boundary[:, 0])
    y0, y1 = np.min(data.boundary[:, 1]), np.max(data.boundary[:, 1])
    data_js['bbxarea'] = float((x1 - x0) * (y1 - y0))
    return HttpResponse(json.dumps(data_js), content_type="application/json")


def GraphSearch(request):
    s=time.perf_counter()
    # Graph
    Searchtype = ["BedRoom", "Bathroom", "Kitchen", "Balcony", "Storage"]
    BedRoomlist = ["MasterRoom", "SecondRoom", "GuestRoom", "ChildRoom", "StudyRoom"]
    NewGraph = json.loads(request.GET.get("NewGraph"))
   
    testname = request.GET.get("userRoomID")
    newNode = NewGraph[0]
    newEdge = NewGraph[1]
    r_Num = np.zeros((1, 14)).tolist()
    r_Mask = np.zeros((1, 14)).tolist()
    r_Acc = np.zeros((1, 14)).tolist()
    r_Num[0][0] = 1
    r_Mask[0][0] = 1
    r_Acc[0][0] = 1

    for indx, rmname, x, y, scalesize in newNode:
        r_Num[0][mdul.vocab['object_name_to_idx'][rmname]] = r_Num[0][mdul.vocab['object_name_to_idx'][rmname]] + 1
        r_Mask[0][mdul.vocab['object_name_to_idx'][rmname]] = 1
        if rmname in BedRoomlist:
            r_Num[0][13] = r_Num[0][13] + 1
            r_Mask[0][13] = 1

    test_index = testNameList.index(testname.split(".")[0])
    topkList = []
    topkList.clear()
    data = test_data[test_index]
   
    Numrooms = json.loads(request.GET.get("Numrooms"))
    

    roomactarr = Numrooms[0]
    roomexaarr = Numrooms[1]
    roomnumarr = [int(x) for x in Numrooms[2]]
    test_data_topk=np.arange(0,74995)

    if np.sum(roomactarr) != 1 or np.sum(roomexaarr) != 1 or np.sum(roomnumarr) != 1:
        test_num = train_data_rNum[test_data_topk]
        # Number filter
     
        filter_func = get_filter_func(roomactarr, roomexaarr, roomnumarr)
        indices = np.where(list(map(filter_func, test_num)))
        # print("np.where(list(map(fil", test_num)
        indices = list(indices)
        test_data_topk = test_data_topk[indices[0]]

    test_num = train_data_eNum[test_data_topk]
    # Graph filter
    
    edgematrix = np.zeros((5, 5))
    for indx1, indx2 in newEdge:
        tmp1 = ""
        tmp2 = ""
        for indx, rmname, x, y, scalesize in newNode:
            if indx1 == indx:
                if rmname in BedRoomlist:
                    tmp1 = "BedRoom"
                else:
                    tmp1 = rmname
        for indx, rmname, x, y, scalesize in newNode:
            if indx2 == indx:
                if rmname in BedRoomlist:
                    tmp2 = "BedRoom"
                else:
                    tmp2 = rmname
        if tmp1 != "" and tmp2 != "":
            edgematrix[Searchtype.index(tmp1)][Searchtype.index(tmp2)] = edgematrix[Searchtype.index(tmp1)][
                                                                             Searchtype.index(tmp2)] + 1
            edgematrix[Searchtype.index(tmp2)][Searchtype.index(tmp1)] = edgematrix[Searchtype.index(tmp2)][
                                                                             Searchtype.index(tmp1)] + 1
    edge = edgematrix.reshape((1, 25))
    filter_graphfunc = filter_graph(edge)
    # rNum_list
    eNumData = []
   
    indices = np.where(list(map(filter_graphfunc, test_num)))

    indices = list(indices)
    tf_trainsub=tf_train[test_data_topk[indices[0]]]
    re_data = train_data[test_data_topk[indices[0]]]
    test_data_tftopk=retrieve_bf(tf_trainsub, data, k=20)
    re_data=re_data[test_data_tftopk]
    if len(re_data) < 20:
        topk = len(re_data)
    else:
        topk = 20
    topkList = []
    for i in range(topk):
        topkList.append(str(re_data[i].name) + ".png")
        
    e=time.perf_counter()
    print('Graph Search time: %s Seconds' % (e - s))

    print("topkList", topkList)
    return HttpResponse(json.dumps(topkList), content_type="application/json")


def retrieve_bf(tf_trainsub, datum, k=20):
    # compute tf for the data boundary
    x, y = rt.compute_tf(datum.boundary)
    y_sampled = rt.sample_tf(x, y, 1000)
    dist = np.linalg.norm(y_sampled - tf_trainsub, axis=1)
    if k > np.log2(len(tf_trainsub)):
        index = np.argsort(dist)[:k]
    else:
        index = np.argpartition(dist, k)[:k]
        index = index[np.argsort(dist[index])]
    return index
from PIL import Image, ImageDraw
import numpy as np

def render_mat_file(request):
    rendered_image = None
    plotly_html = None
    filename = None
    error = None

    if request.method == 'POST' and request.FILES.get('mat_file'):
        mat_file = request.FILES['mat_file']
        filename = mat_file.name

        # Save uploaded file
        upload_path = os.path.join(settings.BASE_DIR, 'temp_upload.mat')
        with open(upload_path, 'wb+') as f:
            for chunk in mat_file.chunks():
                f.write(chunk)

        try:
            # Load mat file - try both data structures
            mat_data = sio.loadmat(upload_path, struct_as_record=False, squeeze_me=True)
            data = None  # Initialize data variable
            
            # === OLD 2D FLOORPLAN RENDERING ===
            try:
                if 'data' in mat_data:
                    data = mat_data['data']
                    boundary = np.array(data.boundary)
                    newBox = np.array(data.newBox)
                    order = np.array(data.order, dtype=int) - 1  # Convert to 0-index
                    rType = np.array(data.rType)
                    doors = np.array(data.doors) if hasattr(data, 'doors') else []
                    windows = np.array(data.windows) if hasattr(data, 'windows') else []

                    # Clear any previous plot
                    plt.close('all')

                    # Use original plot_fp
                    ax = plot_fp(boundary, newBox[order], rType[order], doors, windows)

                    # Get the current figure
                    fig = plt.gcf()
                    fig.set_size_inches(10, 10)

                    # Save to BytesIO
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
                    buf.seek(0)

                    # Convert to base64
                    img_str = base64.b64encode(buf.getvalue()).decode()
                    rendered_image = f"data:image/png;base64,{img_str}"

                    plt.close(fig)  # Clean up
            except Exception as e:
                print(f"2D rendering error: {e}")
                # Try to still get data for 3D
                try:
                    if 'data' in mat_data:
                        data = mat_data['data']
                except:
                    pass

            # === 3D VISUALIZATION (from grok.py) ===
            try:
                # Start with simple grok.py method (most reliable)
                data_3d = sio.loadmat(upload_path)
                record = data_3d["data"][0,0]
                rBound = record["rBoundary"]
                
                # Extract polygons - handle different structures
                try:
                    # Try grok.py method first: rBound[0, i]
                    if rBound.ndim >= 2 and rBound.shape[0] == 1:
                        polys = [rBound[0, i] for i in range(rBound.shape[1])]
                    elif rBound.ndim >= 2:
                        # Try rBound[i] structure
                        polys = [rBound[i] for i in range(len(rBound))]
                    else:
                        # Single polygon
                        polys = [rBound]
                except Exception as poly_extract_error:
                    print(f"Polygon extraction error: {poly_extract_error}")
                    # Last resort: try to extract any way possible
                    if hasattr(rBound, '__iter__'):
                        polys = list(rBound) if len(rBound) > 0 else []
                    else:
                        polys = [rBound] if rBound.size > 0 else []
                
                # Ensure we have polygons
                if len(polys) == 0:
                    raise Exception("No polygons found in rBoundary")
                
                # Single floor color for all rooms
                def get_floor_color(room_idx):
                    """Get floor color - same for all rooms"""
                    return "rgb(220,213,205)"

                # Configuration from grok.py
                wall_height = 25
                wall_thickness = 0.4
                stretch_x = 24
                stretch_y = 10

                # Create 3D figure without doors/windows
                fig = go.Figure()

                # GLOBAL SHADOW BASE - match updated grok.py
                allx = []
                ally = []
                for poly in polys:
                    poly = np.array(poly)
                    
                    # Normalize shapes (match grok.py)
                    if poly.ndim == 3:
                        poly = poly.reshape(-1, 2)
                    if poly.ndim == 1:
                        poly = poly.reshape(-1, 2)
                    if poly.shape[1] != 2:
                        continue
                    
                    xs = poly[:,0] * stretch_x
                    ys = poly[:,1] * stretch_y
                    allx.extend(xs)
                    ally.extend(ys)

                if len(allx) > 0 and len(ally) > 0:
                    xmin, xmax = min(allx), max(allx)
                    ymin, ymax = min(ally), max(ally)

                    fig.add_trace(go.Mesh3d(
                        x=[xmin-10, xmax+10, xmax+10, xmin-10],
                        y=[ymin-10, ymin-10, ymax+10, ymax+10],
                        z=[-4,-4,-4,-4],
                        color='lightgray',
                        opacity=0.20,
                        showlegend=False
                    ))

                # ROOMS - match updated grok.py exactly
                for room_idx, poly in enumerate(polys):
                    try:
                        poly = np.array(poly)
                        
                        # Normalize shape (Nx2 always) - match grok.py
                        if poly.ndim == 3:
                            poly = poly.reshape(-1, 2)
                        if poly.ndim == 1:
                            poly = poly.reshape(-1, 2)
                        if poly.shape[1] != 2:
                            continue
                        
                        xs = poly[:,0] * stretch_x
                        ys = poly[:,1] * stretch_y
                        
                        # Skip degenerate triangles - match grok.py
                        if len(xs) < 3:
                            continue
                        
                        # Skip areas that are too tiny - match grok.py
                        area = abs(np.sum(xs*np.roll(ys,-1) - np.roll(xs,-1)*ys)) / 2
                        if area < 30:  # tweakable threshold
                            continue
                        
                        # Convert to list for plotly - match grok.py
                        xs = xs.tolist()
                        ys = ys.tolist()
                        zs = [0] * len(xs)
                        
                        # FLOOR with room-specific color matching 2D
                        floor_color = get_floor_color(room_idx)
                        
                        # Floor outline (black lines) - match grok.py
                        fig.add_trace(go.Scatter3d(
                            x=xs + [xs[0]],
                            y=ys + [ys[0]],
                            z=zs + [0],
                            mode="lines",
                            line=dict(color="black", width=3),
                            showlegend=False
                        ))
                        
                        # Floor mesh - match grok.py
                        fig.add_trace(go.Mesh3d(
                            x=xs,
                            y=ys,
                            z=np.zeros_like(xs),
                            color=floor_color,
                            opacity=0.95,
                            showlegend=False
                        ))

                        # WALLS - match grok.py
                        for i in range(len(xs)):
                            j = (i+1) % len(xs)

                            # Outward normal extrusion
                            vx = ys[j] - ys[i]
                            vy = xs[i] - xs[j]
                            norm = np.sqrt(vx*vx + vy*vy)
                            if norm > 0:
                                vx /= norm
                                vy /= norm

                            ix1 = xs[i] + vx * wall_thickness
                            iy1 = ys[i] + vy * wall_thickness
                            ix2 = xs[j] + vx * wall_thickness
                            iy2 = ys[j] + vy * wall_thickness

                            # Solid wall panel - match grok.py
                            fig.add_trace(go.Mesh3d(
                                x=[xs[i], xs[j], ix2, ix1],
                                y=[ys[i], ys[j], iy2, iy1],
                                z=[0, 0, wall_height, wall_height],
                                color="white",
                                opacity=1,  # Full solid wall
                                showlegend=False
                            ))

                            # Crisp top outline - match grok.py
                            fig.add_trace(go.Scatter3d(
                                x=[xs[i], xs[j], ix2, ix1, xs[i]],
                                y=[ys[i], ys[j], iy2, iy1, ys[i]],
                                z=[wall_height]*5,
                                mode="lines",
                                line=dict(color="black", width=3),
                                showlegend=False
                            ))
                    except Exception as poly_error:
                        print(f"Error processing polygon {room_idx}: {poly_error}")
                        import traceback
                        traceback.print_exc()
                        continue

                fig.update_layout(
                    scene=dict(
                        aspectratio=dict(x=1, y=1, z=0.20),
                        xaxis_visible=False,
                        yaxis_visible=False,
                        zaxis_visible=False,
                        bgcolor='#1e1e1e'  # Black background like 2D floorplan
                    ),
                    paper_bgcolor='#1e1e1e',  # Black background
                    plot_bgcolor='#1e1e1e',
                    showlegend=False,  # Hide legend
                    autosize=True,
                    width=None,
                    height=None
                )

                plotly_html = fig.to_html(include_plotlyjs='cdn', div_id='plotly-div', config={'responsive': True})

            except Exception as e:
                print(f"3D rendering error: {e}")
                import traceback
                traceback.print_exc()
                # Set error message but don't break the page
                error = f"3D visualization error: {str(e)}" if not error else error

        except Exception as e:
            error = f"Error: {str(e)}"
            import traceback
            traceback.print_exc()

        finally:
            if os.path.exists(upload_path):
                os.remove(upload_path)

    return render(request, "render_mat.html", {
        'rendered_image': rendered_image,
        'plotly_html': plotly_html,
        'filename': filename,
        'error': error,
    })





if __name__ == "__main__":
    pass