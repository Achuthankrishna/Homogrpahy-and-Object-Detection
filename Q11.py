import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

a = cv2.VideoCapture('project2.avi')
x = []
y = []
def houghtrans(frame,res=1,angle_step=1):
    w,h=frame.shape
    teta=np.deg2rad(np.arange(-90,90,angle_step))
    #ggetting diagonal rho
    rhp=np.ceil(int(math.sqrt(w**2+h**2)))
    rhl=np.arange(-rhp,rhp+1,res)
    costeta=np.cos(teta)
    sinteta=np.sin(teta)
    #Create Acccumulator
    acc=np.zeros((len(rhl),len(teta)),dtype=np.uint64)
    yid,xid=np.nonzero(frame)
    for i in range(len(xid)):
        xn=xid[i]
        yn=yid[i]
        for j in range(len(teta)):
            rho=int(rhp+int(xn*costeta[j]+ yn*sinteta[j]))
            acc[rho,j]+=1
    return acc,rhl,teta #getting accumulator score and angles

#Since our application is a video, our hough peaks are not static but forms as a
#parameter in drawing feature lines. So we call hough peaks to adjust peak values
#and accumulator matrix as per frame.


def Hpeaks(H, num_peaks, threshold=300, nhood=2):
     #nhood size is number of which decided the size of the area around a detected peak that should be suppressed
     # #function returns local minima of accumulator matrix as indices of corner points or where lines intersect.
    indicies = []
    for i in range(num_peaks):
    # Find the index of the maximum value in H1
        idx = np.argmax(H)
        H1_idx = np.unravel_index(idx, H.shape)
        indicies.append(H1_idx)
        
     # Suppress peaks in the neighborhood around the maximum value
        y, x = H1_idx
        x_min = max(x - nhood, 0)
        x_max = min(x + nhood + 1, H.shape[1])
        y_min = max(y - nhood, 0)
        y_max = min(y + nhood + 1, H.shape[0])
        H[y_min:y_max, x_min:x_max] = 0

    return indicies, H

def draw_lines(image, index, rho, teta):
    # Define the line drawing color (in RGB)
    color = (0, 0, 255)
    # Loop over the indices and create lines
    for i in range(len(index)):
        r = rho[index[i][0]]
        t = teta[index[i][1]]
        if t==0 and t==np.pi:
             cv2.line(image, (r, 0), (r, image.shape[0]), (255, 0, 0), 2)
        else:
            m = -1 / np.tan(t)
            b = r / np.sin(t)
            y1 = 0
            x1 = int((y1 - b) / m)
            y2 = image.shape[0]
            x2 = int((y2 - b) / m)
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#We need to get corner points of the hough lines
#Since our points are now in hough space we need to find points intersecting the object in normal spcae
# x cos θ1 + y sin θ1 = r1 x cos θ2 + y sin θ2 = r2, in paramterized form and now we need to change it to
# normal form to get points. AX=b , solving this will give x and y .

def hough_inter(img,index,rho,teta):
    cps=[]
    for i in range(len(index)):
        rho1=rho[index[i][0]] #xyz
        teta1=teta[index[i][1]] #teta
        for j in range(i+1,len(index)):
            rho2=rho[index[j][0]]
            teta2=teta[index[j][1]] #teta
            #find solution of form AX=B where A is set of angles and b is  rho in matrix form
            if teta1==teta2 and teta1 != 90+teta2:
                continue
            A=np.array([[np.cos(teta1),np.sin(teta1)],[np.cos(teta2),np.sin(teta2)]])
            # X=[x,y]
            B=np.array([rho1,rho2])
            x,y=np.linalg.solve(A,B)
            if (0 < x < img.shape[1]) and (0 < y < img.shape[0]):
                cps.append((int(x),int(y)))
    return cps

def homography(s,d):
    hmatrix=[]
    for i in range(0, len(s)):
        x, y = s[i][0],s[i][1]
        m, n = d[i][0], d[i][1]
        hmatrix.append([x, y, 1, 0, 0, 0, -m * x, -m * y, -m])
        hmatrix.append([0, 0, 0, x, y, 1, -n * x, -n * y, -n])
    hmatrix = np.array(hmatrix)
    u, s, v= np.linalg.svd(hmatrix)
    Hmatrix = v[-1, :]
    Hmatrix = Hmatrix.reshape(3,3)
    H_norm = Hmatrix/Hmatrix[2,2]
    return H_norm

def decomp(H):
    #Given Intrinsic matrix of the camera
    K = np.array([[0.00138, 0, 0.0946],
                  [0, 0.00138, 0.0527],
                  [0, 0, 1]])
    p=np.array([[0,0,1],[0,279,1],[216,0,1],[279,216,1]])
    #According to formula H= K[R1 R2 R3 T] using this we can calculate R1 R2 R3 and t
    P=np.dot(H,np.linalg.pinv(K))
    r1=P[:,0]/np.linalg.norm(P[:,0],ord=2)
    r2=P[:,1]/np.linalg.norm(P[:,1],ord=2)
    #r3= r1 cross r2
    r3=np.cross(r1,r2)
    t=P[:,2]/np.linalg.norm(P[:,0],ord=2)
    rot=np.column_stack((r1,r2,r3))

    return rot,t



dst_points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
K = np.array([[0.00138, 0, 0.0946],
                  [0, 0.00138, 0.0527],
                  [0, 0, 1]])
while a.isOpened():
    ret, frame = a.read()
    if not ret:
        break
    red = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    red = cv2.erode(red, None, iterations=4)
    red = cv2.dilate(red, None, iterations=4)
    gblur = cv2.GaussianBlur(red, (5, 5), 0)

    red2=cv2.Canny(gblur,70,350)
    
    acc,rho,teta=houghtrans(red2)
    index,acc2=Hpeaks(acc,4, nhood=1)
    #print(index)
    draw_lines(frame,index,rho,teta)
    g=hough_inter(red2,index,rho,teta)
    # print(g)
    n=homography(g,dst_points)
    # print(n)
    rot,trans=decomp(n)
    print("Rotation",rot)
    print("Translation",trans)




    cv2.imshow("lines",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
a.release()
cv2.destroyAllWindows()