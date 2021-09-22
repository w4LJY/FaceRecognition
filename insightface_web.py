from PySide2.QtWidgets import QApplication,QMessageBox,QFormLayout,QGroupBox,QLabel,QWidget,QHBoxLayout,QVBoxLayout,QTextBrowser,QDialog,QToolButton,QFileDialog,QSizePolicy
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import  QIcon,QImage,QPixmap,QMovie
from PySide2.QtCore import QTimer,QSize,Qt,QThread


from insightfaceClass.insightfaceRecognition import Recognition


import numpy as np
import cv2
import sys
import time
import os
import json

class Runthread(QThread):
    #  通过类成员对象定义信号对象
 
    def __init__(self,model):
        super(Runthread, self).__init__()
        self.model = model
 
    def run(self):
        #print('run')
        self.model.embedding_feature()


class labelMe(QDialog):
    def __init__(self,all_ID):
        QDialog.__init__(self)
        self.labelme = QUiLoader().load('ui/labelme.ui')
        self.labelme.CancelButton.clicked.connect(self.labelCancel)
        self.labelme.OkButton.clicked.connect(self.labelOk)
        self.labelme.newlabel.returnPressed.connect(self.labelOk)
        
        self.all_ID = all_ID # 所有ID
        self.returnID = '' # 選擇到的ID
        self.IDtoolButton()
        #self.labelme.finished.connect(self.finish_dialog) # 結束小視窗後觸發
    
    def labelCancel(self):
        self.labelme.close()
    
    def labelOk(self):
        name = self.labelme.newlabel.text()
        if name == '':
            print('未選擇ID')
            QMessageBox.warning (
                self.labelme,
                'warning',
                '\n未選擇ID')
        else:    
            self.returnID = name
            self.labelme.close()
        
    def IDtoolButton(self):
        
        #print('all ID:',self.all_ID)
        scrollWidget = QWidget()   
        VBoxLayout = QVBoxLayout()
        VBoxLayout.setAlignment(Qt.AlignCenter) 
        #VBoxLayout.setContentsMargins(0,0,0,0)
        scrollWidget.setLayout(VBoxLayout)
        self.labelme.labelScroll.setWidget(scrollWidget)
        self.labelme.labelScroll.setWidgetResizable(True)
        for name in self.all_ID:
            toolButton = QToolButton()
            toolbuttonSizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            toolButton.setSizePolicy(toolbuttonSizePolicy)
            toolButton.setText(name)
            toolButton.setAutoRaise(True)
            VBoxLayout.addWidget(toolButton)
            toolButton.clicked.connect(self.run)

    
    def run(self):
        name = self.sender().text()
        self.labelme.newlabel.setText(name)
            
class Web():

    def __init__(self):
        self.ui = QUiLoader().load('ui/main.ui')

        
        #### 背景初始化 ####
        self.setup_bg() 
        
        #### 模型參數初始化 ####
        self.data_path = 'featureID'
        self.data_name = [] # 用來存放偵測到的資料夾
        self.data_init()
        #name = 'fast_det_rec'
        name = 'det_rec'
        #GPU設置
        ctx_id = 0
        #檢測模型參數
        det_size = (640,640)
        det_thresh = 0.5
        self.max_num = 0
        self.model = Recognition(self.ui,name=name,ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        
        self.eid = [] #初始化辨識完成的id
        
        #### 相機初始化 ####
        self.CAM_NUM = 0
        #self.CAM_NUM = 'C:\\Users\\sandman\\Desktop\\PyQt5\\code\\test_pyside2\\insightface_demo\\Muyao4video_long.mp4'
        self.setup_camera() # webwindow載入影像
        
        #### 資訊欄初始化 ####
        self.workinfo_init()
        
        #### 按鍵初始化 ####
        self.ui.closeButton.clicked.connect(self.ui.close)
        self.ui.cameraButton.clicked.connect(self.slotCameraButton)
        self.ui.pauseButton.clicked.connect(self.pauseVideo)
        self.ui.playButton.clicked.connect(self.playVideo)
        self.ui.faceNumGroup.buttonClicked.connect(self.faceNum_switch)
        self.ui.sourceGroup.buttonClicked.connect(self.source_switch)
        self.ui.DataCB.currentIndexChanged.connect(self.changeData)
        
        
        # 滑動窗口
        bar = self.ui.faceScroll.verticalScrollBar()
        bar.rangeChanged.connect(lambda : bar.setValue(bar.maximum()))
        
        self.ui.ClearButton.clicked.connect(self.clear_all)
        self.ui.labelmeButton.clicked.connect(self.popup)
        
        self.thread = Runthread(self.model)
        self.ui.updateButton.clicked.connect(self.updateID)
    
    # def add_face(self):
    #     label1 = QLabel('Slime_%2d' % self.scroll_num)
    #     label2 = QLabel()
    #     label2.setPixmap(QPixmap('bgImg/testImg.png'))
    #     self.formLayout.addRow(label1, label2)
    #     self.scroll_num+=1
        
        
    # def delet_face(self):
    #     if self.formLayout.itemAt(0) is not None:
    #         self.formLayout.itemAt(0).widget().deleteLater()
    #         self.formLayout.itemAt(1).widget().deleteLater()
    
    # def clear_all(self):
    #     for i in range(self.formLayout.count())[:-2]:
    #         self.formLayout.itemAt(i).widget().deleteLater()
    
    def clear_all(self):
        self.eid = []
        
        for i in range(self.formLayout.count()):
            self.formLayout.itemAt(i).widget().deleteLater()
            
        self.ui.scrollBrowser.append('清空檢測名單')
        self.ui.scrollBrowser.ensureCursorVisible()
    
    def popup(self):
        self.pauseVideo()
        #self.ui.sigleFace.setChecked(True)

        face = self.model.catch_IDface(self.frame,self.faces,re_size = False)

        
        dialog = labelMe(self.model.idset)
        ret = dialog.labelme.exec_() # ret回傳值只能傳數字
        if dialog.result() == 0:
            if dialog.returnID == '':
                self.ui.scrollBrowser.append('取消labelImg')
                self.ui.scrollBrowser.ensureCursorVisible()
            else:
                self.ui.scrollBrowser.append('確認labelImg: name = {}'.format(dialog.returnID))
                self.ui.scrollBrowser.ensureCursorVisible()
                self.update_IDdata(dialog.returnID,face[0]) # 雖然臉只會有一張，但是face是個list，裡面的元素才是影像
            self.playVideo() 
    
    
    def update_IDdata(self,img_id,img):
        if img_id not in self.model.idset:
            os.mkdir(os.path.join(self.model.path,img_id))
        
        now_time = time.localtime(int(time.time()))
        timestring = '{}{:0>2d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}'.format(now_time.tm_year,now_time.tm_mon,now_time.tm_mday,now_time.tm_hour,now_time.tm_min,now_time.tm_sec)
        name = '{}_{}.jpg'.format(img_id,timestring)
        cv2.imwrite(os.path.join(self.model.path,img_id,name),img)
    
    def updateID(self):
        #self.pauseVideo()
        #self.thread.start()
        self.ui.updateButton.setEnabled(False)
        self.model.embedding_feature()
        self.ui.scrollBrowser.append('資料庫更新完成')
        self.ui.scrollBrowser.ensureCursorVisible()
        #self.playVideo()
            
    def labelme_flag(self):
        if len(self.faces) == 1:
            self.ui.labelmeButton.setEnabled(True)
        else:
            self.ui.labelmeButton.setEnabled(False)
    
    
        
        
    # 初始化webLabel背景
    def setup_bg(self):
        self.width, self.height = self.ui.webLabel.width(), self.ui.webLabel.height()
        #self.ui.setWindowFlag(Qt.FramelessWindowHint)
        self.ui.centralwidget.setWindowOpacity(0.9) # 設定視窗透明度 
        self.ui.centralwidget.setAttribute(Qt.WA_TranslucentBackground) # 設定視窗背景透明
        
        
        self.web_bg = QMovie('bgImg/load.gif')
        self.web_bg.setScaledSize(QSize(self.width,self.height))
        self.ui.webLabel.setMovie(self.web_bg)
        self.web_bg.start()
        
        self.ui.logoLabel.setPixmap(QPixmap('bgImg/logo.png').scaled(self.ui.logoLabel.size(), Qt.KeepAspectRatio))
        self.ui.faceLabel.setPixmap(QPixmap('bgImg/symbols.png').scaled(QSize(64,64), Qt.KeepAspectRatio))
        # self.ui.faceLabel.setPixmap(QPixmap('C:\\Users\\sandman\\Desktop\\PyQt5\\icon_img\\exe_idea2.png'))
        
        
        self.ui.playButton.setIcon(QIcon('bgImg/play.svg'))
        self.ui.playButton.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.ui.playButton.setAutoRaise(True)
        self.ui.pauseButton.setIcon(QIcon('bgImg/pause.svg'))
        self.ui.pauseButton.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.ui.pauseButton.setAutoRaise(True)

        self.ui.labelmeButton.setIcon(QIcon('bgImg/labels.svg'))

        self.ui.updateButton.setIcon(QIcon('bgImg/refresh.svg'))

        self.ui.ClearButton.setIcon(QIcon('bgImg/trash.svg'))
        self.ui.closeButton.setIcon(QIcon('bgImg/logout.svg'))
        
        self.ui.webRButton.setIcon(QIcon('bgImg/webcam.svg'))
        self.ui.videoRButton.setIcon(QIcon('bgImg/video-lesson.svg'))
        self.ui.cameraButton.setIcon(QIcon('bgImg/youtube.svg'))
        self.ui.singleFace.setIcon(QIcon('bgImg/man.svg'))
        self.ui.multiFace.setIcon(QIcon('bgImg/people.svg'))
        
        scrollWidget = QWidget()   
        self.formLayout = QFormLayout()
        scrollWidget.setLayout(self.formLayout)
        
        self.ui.faceScroll.setWidget(scrollWidget)
        self.ui.faceScroll.setWidgetResizable(True)
        self.scroll_num = 0
        
    def workinfo_init(self):
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.workinfo_space)
        self.update_timer.start(30) # info值設太大，會抓不到

    
    def workinfo_space(self):
        if self.cap.isOpened():
            try:
                info_fps = str(self.fps)
                info_faceNum = str(len(self.faces))
            except:
                info_fps = "None"
                info_faceNum = "None"
            
            if self.ui.pauseButton.isEnabled():
                info_player = 'play'
            else:
                info_player = 'pause'
        else:
            info_fps = "None"
            info_faceNum = "None"
            info_player = "None"
            
        
        if self.CAM_NUM == 0:
            info_source = 'web'
        else:
            info_source = os.path.basename(self.CAM_NUM)
        
        folder_feature = self.model.infolder_feature_list
        
        # 字典生成式，查看每個資料夾內圖片數量
        folder_img = {folder: len([_ for _ in os.listdir(os.path.join(self.model.path,folder))
                   if _.endswith('jpg') or _.endswith('jpeg')]) for folder in os.listdir(self.model.path) }
        
        #print("feature: {} img: {}".format(folder_feature,folder_img))
        
        id_text = ""
        
        info_update = 'complete'
        #self.ui.updateButton.setEnabled(False) 
        for name,value in folder_img.items():
            id_text += '{:<6}: <font color=\"#31EEFB\">{:>4}</font> '.format(name,value)
            if name in folder_feature: # 當出現這種情況，會啟用update(在5行後處理)
                if value != folder_feature[name]:
                    info_update = 'not yet'
                    self.ui.updateButton.setEnabled(True) # 當圖片與feature長度不相同，才啟用update按鈕
            else:
                info_update = 'not yet'
                self.ui.updateButton.setEnabled(True) # 資料夾與feature數量不相同，才啟用update按鈕
                
        if len(folder_img) != len(folder_feature): #增刪資料夾的情況
            info_update = 'not yet'
            self.ui.updateButton.setEnabled(True) # 資料夾與feature數量不相同，才啟用update按鈕
                
        self.ui.infoBrowser.clear()
        info_text = 'FPS: <font color=\"#31EEFB\">{:>6}</font> FaceNum: <font color=\"#31EEFB\">{:>6}</font> Source: <font color=\"#31EEFB\">{:>10}</font> \
                    Player: <font color=\"#31EEFB\">{:>6}</font> Update: <font color=\"#31EEFB\">{:>6}</font>'\
                    .format(info_fps,info_faceNum,info_source,info_player,info_update)
        self.ui.infoBrowser.append(info_text)
        class_text = 'Class: <font color=\"#31EEFB\">{:>6}</font>'.format(len(folder_img))
        self.ui.infoBrowser.append(class_text)
        self.ui.infoBrowser.append('')
        self.ui.infoBrowser.append(id_text)
        
        # self.ui.DataCB.clear()
        
        now_folder_name = []
        for folder in os.listdir(self.data_path):
            now_folder_name.append(folder)
            
        if now_folder_name != self.data_name:
            self.ui.scrollBrowser.append('更動群組資料夾')
            self.ui.scrollBrowser.ensureCursorVisible()
            self.ui.DataCB.clear()
            self.ui.DataCB.addItems(now_folder_name)
            self.data_name = now_folder_name
            
    
    def data_init(self):
        for folder in os.listdir(self.data_path):
            self.data_name.append(folder)
            self.ui.DataCB.addItem(folder)
            
    def changeData(self):
        self.model.path = os.path.join(self.data_path,self.ui.DataCB.currentText())
        self.model.embedding_feature()
        self.ui.scrollBrowser.append('類別資料夾更換為: {}'.format(self.ui.DataCB.currentText()))
        self.ui.scrollBrowser.ensureCursorVisible()
        
    # 初始化相機與執行續
    def setup_camera(self):
        self.cap = cv2.VideoCapture() #初始化摄像头
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        
    # Qtimer抓取影像，辨識
    def display_video_stream(self):
        ret,self.frame = self.cap.read()
        #self.frame = self.model.resize_max(self.frame,self.height,self.width) # 外部影片載入用
        if not ret:
            self.ui.scrollBrowser.append('影片播放完畢')
            self.ui.scrollBrowser.ensureCursorVisible()
            self.closeCamera() # 外部影片載入用
            
        if self.CAM_NUM == 0: # web camera才需要翻轉
            #self.frame = cv2.flip(self.frame, 1)
            pass
        
        start = time.time()
        
        ### pred ###
        self.faces = self.model.pred(self.frame,self.max_num)
        self.labelme_flag() # 判斷人臉數量為1，才啟用labelme按鈕
        
        #### catchFace ####
        totalFace = self.model.catch_face(self.frame,self.faces, self.ui.faceLabel.height(),self.ui.faceLabel.width())
        stotalFace = self.model.catch_face(self.frame,self.faces, 100,100)
        
        
        #### draw ####
        self.tframe = self.model.draw(self.frame,self.faces)
        if len(self.faces):
            if self.faces[0].embedding is not None:
                #totalId = self.model.embedding_similarity(self.faces)
                totalId = self.model.total_id
                self.idImg(stotalFace,totalId)
        
       
                
        self.faceImg(totalFace)
          

        end = time.time()
        self.fps_eval(start,end)
        self.webImg()
        
    # fps計算
    def fps_eval(self,start,end):
        # 輸出結果
        self.fps = round(1/max(sys.float_info.epsilon, end-start),2)
        #cv2.putText(self.tframe, 'FPS: '+str(self.fps), (10,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
        #print("FPS：%f " % (self.fps))
    
    # 更新辨識後的畫面webLabel
    def webImg(self):
        self.tframe = cv2.cvtColor(self.tframe, cv2.COLOR_BGR2RGB)
        #print(self.ui.webLabel.width(), self.ui.webLabel.height())
        self.tframe = self.model.face_resize(self.tframe,self.ui.webLabel.height()-10,self.ui.webLabel.width()-10)
        
        showImage = QImage(self.tframe, self.tframe.shape[1],self.tframe.shape[0],self.tframe.strides[0],QImage.Format_RGB888)
        self.ui.webLabel.setPixmap(QPixmap.fromImage(showImage))
        self.web_bg.stop()
    
    # 更新擷取的臉部影像faceLabel
    def faceImg(self,totalFace):
        if totalFace:
            '''
            這裡還沒處理乾淨，同一個frame擷取到多張人臉，只會顯示第一張
            '''
            faceImg = cv2.cvtColor(totalFace[0], cv2.COLOR_BGR2RGB) 
            detFace = QImage(faceImg, faceImg.shape[1],faceImg.shape[0],3*faceImg.shape[1],QImage.Format_RGB888)
            self.ui.faceLabel.setPixmap(QPixmap.fromImage(detFace))
            
    def idImg(self,totalFace,totalId):
        
        #horLayout = QHBoxLayout()
        
        
        nId = [n for n in totalId if n not in self.eid+['unknown']]
        index = [i for i,n in enumerate(totalId) if n not in self.eid+['unknown']]
        self.eid+=nId
        for i in index:
            label1 = QLabel()
            faceImg = cv2.cvtColor(totalFace[i], cv2.COLOR_BGR2RGB) 
            detFace = QImage(faceImg, faceImg.shape[1],faceImg.shape[0],3*faceImg.shape[1],QImage.Format_RGB888)
            label1.setPixmap(QPixmap.fromImage(detFace))
            label1.setStyleSheet('''QLabel{padding:2px 2px 2px 2px;}QLabel{border-radius:4px; }QLabel{border-radius:10px; }QLabel{border:4px solid #2196f3;};''')
            
	
            #label2 = QLabel('({}) id = {}'.format(self.scroll_num,totalId[i]))
            label_text = '  id = {}'.format(totalId[i])
            
            sex = ''
            info_json_path = os.path.join(self.model.path, totalId[i], 'name.json')
            if os.path.exists(info_json_path):
                with open(info_json_path) as info_json:
                    info = json.load(info_json)
                    
                for Attribute,value in info.items():
                    if Attribute == 'sex':
                        sex = value
                    label_text += '\n  {} = {}'.format(Attribute,value)
            else:
                label_text+= '\n  sex = ?\n  age = ?'
                
            
            
            label2 = QLabel(label_text)
            if sex == 'Male':
                label2.setStyleSheet('''QLabel{background:#6DDF6D;border-radius:5px;}QLabel{height:100px};''')
            elif sex == 'Female': 
                label2.setStyleSheet('''QLabel{background:#F76677;border-radius:5px;}QLabel{height:100px};''')
            else:
                label2.setStyleSheet('''QLabel{background:#F7D674;border-radius:5px;}QLabel{height:100px};''')
            
           

            self.formLayout.addRow(label1, label2)
            self.scroll_num+=1
            # imgLabel = QLabel()
            # faceImg = cv2.cvtColor(totalFace[i], cv2.COLOR_BGR2RGB) 
            # detFace = QImage(faceImg, faceImg.shape[1],faceImg.shape[0],3*faceImg.shape[1],QImage.Format_RGB888)
            # imgLabel.setPixmap(QPixmap.fromImage(detFace))
            # IDBrowser = QTextBrowser()
            # IDBrowser.append('({:>3d}) id = {}'.format(self.scroll_num,totalId[i]))
            
            # horLayout.addWidget(imgLabel)
            # horLayout.addWidget(IDBrowser)
            # self.formLayout.addrow(imgLabel,IDBrowser)
            # self.scroll_num+=1
            
            
        
    # 影像顯示關閉
    def slotCameraButton(self):
        if self.timer.isActive() == False and not(self.cap.isOpened()): # 加上isOpened防止暫停誤判成影像已關閉
            self.openCamera()
        else:
            self.closeCamera() 

    def openCamera(self):
        flag = self.cap.open(self.CAM_NUM)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width) # 沒什麼意義
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if flag == False:
            QMessageBox.critical(
                self.ui,
                'Error',
                '請檢查電腦與相機是否正確連接')
            print('請檢查電腦與相機是否正確連接')
        else:
            self.timer.start(30)
            self.ui.cameraButton.setText(' close')
            self.ui.cameraButton.setIcon(QIcon('bgImg/stop.svg'))
            self.ui.pauseButton.setEnabled(True)
            #self.ui.updateButton.setEnabled(True)
            self.ui.webRButton.setEnabled(False)
            self.ui.videoRButton.setEnabled(False)
    
    def closeCamera(self):
        self.timer.stop()
        self.cap.release()
        self.ui.webLabel.setMovie(self.web_bg)
        self.web_bg.start()
        self.ui.cameraButton.setText(' show')   
        self.ui.faceLabel.setPixmap(QPixmap('bgImg/symbols.png').scaled(QSize(64,64), Qt.KeepAspectRatio))
        self.ui.cameraButton.setIcon(QIcon('bgImg/youtube.svg'))
        self.ui.pauseButton.setEnabled(False)
        self.ui.playButton.setEnabled(False)
        self.ui.labelmeButton.setEnabled(False)
        self.ui.updateButton.setEnabled(False) # update本身是不會出問題，但暫停播放就會受影響
        self.ui.webRButton.setEnabled(True)
        self.ui.videoRButton.setEnabled(True)
        
    def pauseVideo(self):
        self.timer.stop()
        self.ui.pauseButton.setEnabled(False)
        self.ui.playButton.setEnabled(True)
            
    def playVideo(self):
        self.timer.start(30)
        
        self.ui.playButton.setEnabled(False)
        self.ui.pauseButton.setEnabled(True)

    # 切換檢測到的人臉數上限
    def faceNum_switch(self,id):
        check_text=self.ui.faceNumGroup.checkedButton().text()
        if check_text == ' single':
            self.max_num = 1
        else:
            self.max_num = 0
    
    # 切換輸入形式(web,video)，顯示影像前可調整
    def source_switch(self):
        check_text=self.ui.sourceGroup.checkedButton().text()
        if check_text == ' web':
            self.CAM_NUM = 0
            
        elif check_text == ' video':
            filePath, _  = QFileDialog.getOpenFileName(
                        self.ui,
                        "選擇你要上傳的影片", # 標題
                        r"G:\\我的雲端硬碟\\ML\\Colab Notebooks\\Summer_Internship\\recognition\\Inference_code\\datasets\\Muyao4_test",        # 起始目錄
                        "圖片類型 (*.mp4 *.avi)" #通過副檔名過濾文件
                    )
            self.CAM_NUM = filePath
            if filePath == '':
                self.ui.scrollBrowser.append('取消選取影片')
                self.ui.scrollBrowser.ensureCursorVisible()
                self.CAM_NUM = 0
                self.ui.webRButton.setChecked(True)
        

if '__main__' == __name__:
    app = QApplication(sys.argv)
    #app.setWindowIcon(QIcon('insightface_logo.png'))
    webwindow = Web()
    webwindow.ui.show()
    sys.exit(app.exec_())
