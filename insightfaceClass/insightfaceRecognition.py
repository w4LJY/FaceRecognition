import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os,json
from sklearn.metrics import pairwise # 做人臉相似度比對
import shutil

# 讓cv2.imread能讀中文路徑
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img
    
class Recognition(FaceAnalysis):
    
    def __init__(self, ui, name='antelopev2', ctx_id = 0, det_size = (640,640), det_thresh = 0.5):
        super().__init__(name=name,root='insightface_onnx')
        self.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        
        self.ui = ui
        self.json_file = 'embeddings_r18.json'
        resize_img_flag = False
        #self.path = 'featureID/Muyao4'
        self.path = os.path.join('featureID',self.ui.DataCB.currentText())
        print('群組資料夾為 = ',self.path)
        self.dist_th = 0.35
        
         
        if 'recognition' in self.models:
            self.embedding_feature(resize_img_flag=resize_img_flag) # 初始化/更新 face_db
    
    def embedding_feature(self, resize_img_flag = False):
    
        self.face_db = {} # 儲存embedding的人臉資料庫
        self.infolder_feature_list = {} # 儲存每個ID的feature長度，用在之後判斷是否要更新資料庫
        
        self.idset = set() # 存放所有身分
        for folder in os.listdir(self.path):
          
          self.idset.add(folder)
          folderjoin = os.path.join(self.path, folder) 
          if os.path.isdir(folderjoin):
            self.face_db[folder] = {"embeddings":[]} 
            infolder_img_number = len([_ for _ in os.listdir(folderjoin) if _.endswith('jpg') or _.endswith('jpeg')])
            
        
            if self.json_file in os.listdir(folderjoin):
              with open(os.path.join(self.path, folder, self.json_file)) as json_data_file:
                try:
                  data = json.load(json_data_file)
                  self.face_db[folder]["embeddings"] = data["embeddings"]
                  infolder_feature_number = len(self.face_db[folder]["embeddings"])
                  
                except Exception as e: 
                  pass # 這裡根本不會進來，不知道幹嘛用的
                else: 
                  if infolder_img_number == infolder_feature_number:
                      self.ui.scrollBrowser.append("skip folder: {}".format(folder))
                      self.ui.scrollBrowser.ensureCursorVisible()
                      print("skip folder: {} (n={})".format(folder,infolder_img_number))  
                      self.infolder_feature_list[folder] = infolder_feature_number #儲存每個ID的feature長度 - 沒有變動過的
                      continue

            if self.face_db[folder]["embeddings"] == []:
                infolder_feature_number = 0 # 新增的資料夾，原本feature是0(原本空的list長度也會是1)
            self.ui.scrollBrowser.append("folder: {}".format(folder))
            self.ui.scrollBrowser.ensureCursorVisible()
            #print("folder: {} (n={},f={})".format(folder,infolder_img_number,infolder_feature_number))
            self.face_db[folder]["embeddings"] = []
            
            face_add_num = 0 #計算新增了幾張人臉
            for filename in os.listdir(folderjoin):
              filenamejoin = os.path.join(self.path, folder, filename) 
        
              ext = os.path.splitext(filenamejoin)[-1].lower() 
              if ext == '.jpg' or ext == '.jpeg':  #資料庫圖片最好不要丟png，有時候會抱錯
                face_add_num+=1
                #print('readimg')
                img = cv_imread(filenamejoin)
                if resize_img_flag:
                    img = self.resize_max(img, 480, 640) 
                    print('resize')
        
                face_data = self.get(img,max_num=1) # 這裡會限制最多檢測1人，多人檢測下方還是會報錯  
        
                rimg = self.draw_on(img, face_data)
                # cv2.imshow('embedding feature', rimg)
                # cv2.waitKey(0)
        
                if len(face_data) != 1: 
                    self.ui.scrollBrowser.append("less face, remove: {}".format(filenamejoin))
                    self.ui.scrollBrowser.ensureCursorVisible()
                    print("!!!less or more then one face found in picture!!! remove: ",filenamejoin)
                    os.remove(filenamejoin)
                    if infolder_img_number == 1:
                        shutil.rmtree(folderjoin)
                        self.idset.remove(folder)
                        del self.face_db[folder]
                        print('未新增類別')
                    face_add_num-=1
                    continue
                    #exit()
                self.face_db[folder]["embeddings"].append(face_data[0].embedding.tolist()) 
            
            if os.path.isdir(folderjoin): # 如果資料夾被刪空，用這條防止出錯
              if face_add_num - infolder_feature_number>0:
                change_text = '新增'
              else:
                change_text = '移除'
              self.ui.scrollBrowser.append('{}{}張人臉特徵'.format(change_text,abs(face_add_num - infolder_feature_number)))
              self.ui.scrollBrowser.ensureCursorVisible()
              print('新增{}張人臉特徵'.format(face_add_num - infolder_feature_number))  # 全部的人臉(自動刪除(沒臉或多臉)不算) - 原來就有的人臉
              self.infolder_feature_list[folder] = len(self.face_db[folder]["embeddings"]) #儲存每個ID的feature長度 - 變動過的
            
              with open(os.path.join(self.path, folder, self.json_file), 'w') as outfile:
                json.dump(self.face_db[folder], outfile)
        
        self.ui.scrollBrowser.append("--------------------------")
        self.ui.scrollBrowser.ensureCursorVisible()
    # resize資料集的人臉，加快運行速度
    def resize_max(self, img, max_h, max_w):
        #resize image to improve interference speed
        (h, w) = img.shape[:2]
        if h > max_h:
            height = max_h
            width = w*max_h/h
            img = cv2.resize(img, (int(width),int(height)))
        (h, w) = img.shape[:2]
        if w > max_w:
            height = h*max_w/w
            width = max_w
            img = cv2.resize(img, (int(width),int(height)))
        #print('new',img.shape[:2])
        return img
    
    def pred(self, img, max_num = 0):
        faces = self.get(img, max_num = max_num)
        return faces
    
    def draw(self, img, faces):
        
        self.total_id = []
        timg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            
            box = face.bbox.astype(np.int)
            color = (0, 255, 0)
            cv2.rectangle(timg, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            if face.kps is not None:
              kps = face.kps.astype(np.int)
              for j in range(len(kps)):
                color = (0, 255, 255)
                cv2.circle(timg, (kps[j][0], kps[j][1]), 1, color, 5)
            
            if face.landmark_2d_106 is not None:
                landmark = np.round(face.landmark_2d_106).astype(np.int) # landmark_3d_68, landmark_2d_106
                for k in range(len(landmark)):
                  # 畫成漂亮的顏色
                  if k < 33: # 臉頰
                    color = (106, 18, 156)
                  elif k < 43: # 右眼
                    color = (0, 255, 255)
                  elif k < 52: # 右眉
                    color = (255, 26, 5)
                  elif k < 72: # 嘴巴
                    color = (1, 31, 255)
                  elif k < 87: # 鼻子
                    color = (80, 176, 0)
                  elif k < 97: # 左眼
                    color = (0, 185, 255)
                  else:      # 左眉
                    color = (252, 169, 19)
                  cv2.circle(timg, (landmark[k][0], landmark[k][1]), 1, color, 2)
                  
            if face.gender is not None and face.age is not None:
                cv2.putText(timg,'%s,%d'%(face.sex,face.age), (box[2]-40, box[1]-4),cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
            if face.embedding is not None:
                name_list = []
                dist_list = []
                for name, value in self.face_db.items(): 
                    #print("db: {} face: {}".format(np.array(value["embeddings"]).shape,np.array([face.embedding]).shape))
                    dist = np.max(pairwise.cosine_similarity(value["embeddings"], [face.embedding]))
                    if dist > self.dist_th: 
                        name_list.append(name)
                        dist_list.append(dist)
                  
                if name_list:
                    max_index = dist_list.index(max(dist_list))
                    self.id = name_list[max_index]
                else:
                    self.id = 'unknown'
                self.total_id.append(self.id)
                cv2.putText(timg, self.id, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 1, cv2.LINE_AA) # 顯示該人臉的身分類別

        return timg
    
    def embedding_similarity(self, faces):
        total_id = []
        for i in range(len(faces)):
            face = faces[i]
            
            name_list = []
            dist_list = []
            for name, value in self.face_db.items(): 
                dist = np.max(pairwise.cosine_similarity(value["embeddings"], [face.embedding]))
                if dist > self.dist_th: 
                    name_list.append(name)
                    dist_list.append(dist)
              
            if name_list:
                max_index = dist_list.index(max(dist_list))
                self.id = name_list[max_index]
            else:
                self.id = 'unknown'
            total_id.append(self.id)    
            
            return total_id
    
    def catch_face(self, img, faces, maxH, maxW):
        
        total_face = []
        for i in range(len(faces)):
            face = faces[i]
            
            box = face.bbox.astype(np.int)
            box[0],box[2] = np.clip((box[0],box[2]),0,img.shape[1])
            box[1],box[3] = np.clip((box[1],box[3]),0,img.shape[0])
            faceImg = img[box[1]:box[3],box[0]:box[2]]
            
            faceImg = self.face_resize(faceImg, maxH, maxW)
            total_face.append(faceImg)
        
        return total_face
    
    def catch_IDface(self, img, faces, maxH=256, maxW=256,re_size=False):
        total_face = []
        for i in range(len(faces)):
            face = faces[i]
            
            box = face.bbox.astype(np.int)
            
            # 擴大bbox以將捕獲的圖像用作人臉數據庫
            per_x = int((box[2] - box[0]) * 0.35)
            per_y = int((box[2] - box[0]) * 0.35)
            box = [box[0] - per_x, box[1] - per_y, box[2] + per_x, box[3] + per_y]
            
            box[0],box[2] = np.clip((box[0],box[2]),0,img.shape[1])
            box[1],box[3] = np.clip((box[1],box[3]),0,img.shape[0])
            faceImg = img[box[1]:box[3],box[0]:box[2]]
            
            if re_size:
                faceImg = self.face_resize(faceImg, maxH, maxW)
            total_face.append(faceImg)
        
        return total_face
            
            
    def face_resize(self, im, maxH, maxW):
        if im.shape[0] > im.shape[1]:
            height = maxH
            width = im.shape[1]*maxH/im.shape[0]
        else:
            height = im.shape[0]*maxW/im.shape[1]
            width = maxW
        
        return cv2.resize(im, (int(width),int(height)))
