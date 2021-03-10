import cv2
import mediapipe as mp
import numpy as np
from test_hands_pose_ import detect,hand_pose_label
from playsound import playsound
import os
import threading
import time

voice_flog =False
display_count = 1

threshold_count=18

reg_up=0
reg_num=[0,0,0,0,0]
c_count=[0,0,0,0,0]

def to_zero():
    global c_count
    for i in range(5):
        c_count[i]=0
        

def play_voice():
    global display_count
    playsound('./voice/{}.mp3'.format(display_count))

def compute_display(ans):
    
    
    
    up ="good"
    #_num=["one","two","three","four"]
    num={"one":1,"two":2,"three":3,"four":4}
    
    global display_count,reg_up,reg_num,c_count
    
    
    if(ans==up):
        c_count[0]+=1
        if(c_count[0]>threshold_count and reg_up==1):
            display_count+=1
            to_zero()
            reg_up=0
    elif(ans!=up and reg_up==0):
        reg_up=1
    
    if(ans=="one"):
        c_count[1]+=1
        if(c_count[1]>threshold_count and reg_num[1]==1):
            display_count=1
            to_zero()
            reg_num[1]=0
    elif(ans!="one" and reg_num[1]==0):
        reg_num[1]=1
        
    if(ans=="two"):
        c_count[2]+=1
        if(c_count[2]>threshold_count and reg_num[2]==1):
            display_count=2
            to_zero()
            reg_num[2]=0
    elif(ans!="two" and reg_num[2]==0):
        reg_num[2]=1
    
    if(ans=="three"):
        c_count[3]+=1
        if(c_count[3]>threshold_count and reg_num[3]==1):
            display_count=3
            to_zero()
            reg_num[3]=0
    elif(ans!="three" and reg_num[3]==0):
        reg_num[3]=1
    
    if(ans=="four"):
        c_count[4]+=1
        if(c_count[4]>threshold_count and reg_num[4]==1):
            display_count=4
            to_zero()
            reg_num[4]=0
    elif(ans!="four" and reg_num[4]==0):
        reg_num[4]=1
    
    # if(ans==up and reg_up==1):
    #     #display_count+=1
    #     c_count[0]+=1
    #     reg_up=0
    # elif(ans!=up and reg_up==0):
    #     reg_up=1
    
    
    #print(reg_num)
    
    #if(ans in num):
    #     val = num[ans]
    #     if(reg_num[val]==1):
    #         display_count=val
    #         #c_count[val]+=1
    #         reg_num[val]=0
    # # elif(ans =="ok"):
    # #   for i in range(5):reg_num[i]=1
    #     for i in num:
    #          if( i!=ans and reg_num[num[i]]==0):
    #              reg_num[num[i]]=1       
    # else:
    #     for i in range(5):reg_num[i]=1
    
    
    # if(c_count[0]>threshold_count):
    #     display_count+=1
    #     to_zero()
    # for i ,val in enumerate(c_count):
    #     if(i!=0 and val>threshold_count):
    #         display_count=val
    #         to_zero()
            
    
    
    if(display_count>4):
        display_count=1
        
    


def get_text_list():
    H,W=88,72
    cat_img=None
    cat_text=None
    for i,name in enumerate(hand_pose_label):
        img=cv2.imread("./similar_test/{}.jpg".format(name))
        re_img=cv2.resize(img, (W,H))
        text_img = np.zeros((H, W, 3), np.uint8)
        text_img.fill(90)
        cv2.putText(text_img,name, (10, 80), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 255), 1, cv2.LINE_AA)
        if(i==0):
            cat_img=re_img.copy()
            cat_text=text_img.copy()
        else:
            cat_img = cv2.hconcat([cat_img,re_img])
            cat_text = cv2.hconcat([cat_text,text_img])
    cat_all=cv2.vconcat([cat_text,cat_img])
    #cv2.imshow('show',cat_all)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return cat_all


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5,max_num_hands=1)


if __name__ == '__main__':
    
    
    
    reg_dis=[0,0,0,0,0]
    display_img=[0]
    for img in range(len(os.listdir("./display"))):
        disp_img=cv2.imread("./display/{}.jpg".format(img+1))
        disp_img =cv2.resize(disp_img,(960,1280))
        display_img.append(disp_img)    
    
    cap = cv2.VideoCapture(0)
    text_img=get_text_list()[60:-1,:]
    
    #voice = threading.Thread(target = play_voice)
 
    while cap.isOpened():
        
      #print(c_count)
        
      success, image = cap.read()
      
      if not success:
        print("cap:{}".format(success))
        break
    
      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = hands.process(image)
    
      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      ccc_img = np.zeros(image.shape, np.uint8)
      if results.multi_hand_landmarks:
        #print('handedness:', results.multi_handedness)
        for hand_landmarks in results.multi_hand_landmarks:
          # mp_drawing.draw_landmarks(
          #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
          mp_drawing.draw_landmarks(
              ccc_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
          
      gray = cv2.cvtColor(ccc_img, cv2.COLOR_BGR2GRAY)
      cnts,_= cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      clone = image.copy()
      cccc_img=ccc_img.copy()
      for c in cnts:
          (x, y, w, h) = cv2.boundingRect(c)
          #print(w,h)
          cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 0, 255), 3)
          if(w>=180 or h>=180):
              Ans=detect(ccc_img[y:y+h,x:x+w])
              cccc_img = cv2.putText(cccc_img, Ans, (x,y), 1, 3, (0, 0, 255), 3)
              compute_display(Ans)
          else:
              to_zero()
              reg_up=1
              for i in range(5):reg_num[i]=1
              
      
      #0
      result_img = np.zeros((360+100+1280,960, 3), np.uint8)
      #1
      clone = cv2.resize(clone,(480,360))
      cccc_img = cv2.resize(cccc_img,(480,360))
      cat_all=cv2.hconcat([clone,cccc_img])
      result_img[0:360,0:960]=cat_all
      #2
      text_img=get_text_list()[60:-1,:]
      text_img=cv2.resize(text_img,(960,100))
      result_img[360:360+100,0:960]=text_img
      #3
      result_img[360+100:360+480+1280,0:960]=display_img[int(display_count)]
      
      for i ,val in enumerate(reg_dis):
          if(i==display_count and reg_dis[i]==1):
              threading.Thread(target = play_voice).start()
              reg_dis[i]=0
          elif(i!=display_count and reg_dis[i]==0):
              reg_dis[i]=1

      
      if(voice_flog):
          threading.Thread(target = play_voice).start()
         
      cv2.namedWindow('show', cv2.WINDOW_NORMAL)
      cv2.imshow('show', result_img)
      
      
      if cv2.waitKey(1) & 0xFF == 27:
        break
      
    
    cv2.destroyAllWindows()
    hands.close()
    cap.release()