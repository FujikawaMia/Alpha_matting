import cv2 as cv
from math import sqrt
from math import ceil
from math import cos
from math import sin
from math import exp
from math import pi
import numpy as np
import time
from numba import cuda
from numba import njit
from numba.typed import List

@cuda.jit
def Dp_cuda(x_1,y_1,x_2,y_2):
    return sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)

@njit
def Dp(x_1,y_1,x_2,y_2):
    return sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)


@cuda.jit
def Dc_cuda(tuple1, tuple2):
    return (tuple1[0]-tuple2[0])**2 + (tuple1[1]-tuple2[1])**2 + (tuple1[2]-tuple2[2])**2

@njit
def Dc(tuple1, tuple2):
    return (tuple1[0]-tuple2[0])**2 + (tuple1[1]-tuple2[1])**2 + (tuple1[2]-tuple2[2])**2


@njit
def pfP(x, y, f, b, image):
    fmin = 1E10
    for i in range(len(f)):
        fp = eP(x, y, f[i][0], f[i][1], image)
        if (fp < fmin):
            fmin = fp
    
    bmin = 1E10
    for i in range(len(b)):
        bp = eP(x, y, b[i][0], b[i][1], image)
        if (bp < bmin):
            bmin = bp
        
    return bmin / (fmin + bmin + 1E-10)


@njit
def eP(i1, j1, i2, j2, image):
    ci = i2 - i1
    cj = j2 - j1
    z  = sqrt(ci * ci + cj * cj)

    ei = ci / (z + 1E-6)
    ej = cj / (z + 1E-6)

    step = min(1 / (abs(ei) + 1E-10), 1 / (abs(ej) + 1E-10))

    result = 0

    pre = image[i1][j1]

    ti = i1
    tj = j1

    t = 1
    while True:

        inci = ei * t
        incj = ej * t

        if(abs(ci) <= abs(inci) or abs(cj) <= abs(incj)):
            break
    
        i = int(i1 + inci + 0.5)
        j = int(j1 + incj + 0.5)

        z = 1
        cur = image[i][j]

        if (ti - i > 0 and tj - j == 0):
            z = abs(ej)
        elif(ti - i == 0 and tj - j > 0):
            z = abs(ei)

        result += ((cur[0] - pre[0])**2 +
                  (cur[1] - pre[1])**2 + 
                  (cur[2] - pre[2])**2) * z
        pre = cur

        ti = i
        tj = j

        t += step

    return result


@njit
def gP(x, y, f_x, f_y, b_x, b_y, pf, dpf, height, width, image):
    f = image[f_x, f_y]
    b = image[b_x, b_y]
    dpb = Dp(x, y, b_x, b_y)

    tn = nP(x, y, f, b, height, width, image) 
    ta = aP(x, y, pf, f, b ,image) ** 2

    return (tn ** 3) * (ta ** 2) * dpf * (dpb ** 4)


@njit
def aP(x, y, pf, f, b, image):
    p = image[x, y]
    alpha = getalpha(p, f, b)

    return pf + (1 - 2 * pf) * alpha


@njit
def getalpha(p, f, b):
    alpha = (((p[0] - b[0]) * (f[0] - b[0]) + 
              (p[1] - b[1]) * (f[1] - b[1]) + 
              (p[2] - b[2]) * (f[2] - b[2]))
            /((f[0] - b[0]) ** 2 + 
              (f[1] - b[1]) ** 2 + 
              (f[2] - b[2]) ** 2 + 1E-6))
    return min(1, max(0, alpha))


@njit
def nP(x, y, f, b, height, width, image):
    i1 = max(0, x - 1)
    i2 = min(x + 1, height - 1)
    j1 = max(0, y - 1)
    j2 = min(y + 1, width - 1)

    result = 0

    for k in range(i1, i2 + 1):
        for l in range(j1, j2 + 1):
            m = mP(k, l, f, b, image)
            result += m * m

    return result


@njit
def mP(i, j, f, b, image):
    p = image[i][j]
    alpha = getalpha(p, f, b)

    result = sqrt((p[0] - alpha * f[0] - (1 - alpha) * b[0]) * (p[0] - alpha * f[0] - (1 - alpha) * b[0]) +
                    (p[1] - alpha * f[1] - (1 - alpha) * b[1]) * (p[1] - alpha * f[1] - (1 - alpha) * b[1]) +
                    (p[2] - alpha * f[2] - (1 - alpha) * b[2]) * (p[2] - alpha * f[2] - (1 - alpha) * b[2]))
    return result / 255


@njit
def sigma_squre(tf, height, width, image):
    x = tf[0]
    y = tf[1]
    pc = image[x][y]

    i1 = max(0, x - 2)
    i2 = min(x + 2, height - 1)
    j1 = max(0, y - 2)
    j2 = min(y + 2, width - 1)

    result = 0
    num = 0

    for i in range(i1, i2+1):
        for j in range(j1, j2+1):
            temp = image[i][j]
            result += Dc(pc, temp)
            num = num + 1

    return result / (num + 1E-10)


@cuda.jit
def expansion(image,trimap,new_trimap,height,width):
  KI = 10
  KC = 5 * 5
  tx = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
  ty = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y

  if tx < height and ty < width:
    if (trimap[tx][ty] != 0 and trimap[tx][ty] != 255):
        flag = False
        
        for k in range (KI + 1):
            if flag is False:
                k1 = max(0, tx - k)
                k2 = min(tx + k, height - 1)
                l1 = max(0, ty - k)
                l2 = min(ty + k, width - 1)

                for l in range(k1, k2 + 1):
                    if flag is False:
                        gray = trimap[l][l1]
                        if (gray == 0 or gray == 255):
                            dis = Dp_cuda(tx, ty, l, l1)

                            if (dis > KI):
                                continue

                            distanceColor = Dc_cuda(image[tx][ty], image[l][l1])
                            if (distanceColor <= KC):
                                flag = True
                                new_trimap[tx][ty] = gray
                            
                        if (flag):
                            break

                        gray = trimap[l][l2]
                        if (gray == 0 or gray == 255):
                            dis = Dp_cuda(tx, ty, l, l2)

                            if (dis > KI):
                                continue

                            distanceColor = Dc_cuda(image[tx][ty], image[l][l2])
                            if (distanceColor <= KC):
                                flag = True
                                new_trimap[tx][ty] = gray
                            
                        if (flag):
                            break
                    else:
                        break


                for l in range(l1, l2 + 1):
                    if flag is False:
                        gray = trimap[k1][l]
                        if (gray == 0 or gray == 255):
                            dis = Dp_cuda(tx, ty, k2, l)

                            if (dis > KI):
                                continue

                            distanceColor = Dc_cuda(image[tx][ty], image[k1][l])
                            if (distanceColor <= KC):
                                flag = True
                                new_trimap[tx][ty] = gray
                            
                        if (flag):
                            break

                        gray = trimap[k2][l]
                        if (gray == 0 or gray == 255):
                            dis = Dp_cuda(tx, ty, k1, l)

                            if (dis > KI):
                                continue

                            distanceColor = Dc_cuda(image[tx][ty], image[k2][l])
                            if (distanceColor <= KC):
                                flag = True
                                new_trimap[tx][ty] = gray
                            
                        if (flag):
                            break
                    else:
                        break
            else:
                break


@njit
def sampling(F,B,width,height,tp,trimap):
    KG = 4
    a=90
    b=17

    x = tp[0]
    y = tp[1]

    angle=(x+y)*b % a
    for step in range(KG):
        f1 = False
        f2 = False

        z=((angle + step*a)/180) * pi
        ex=sin(z)
        ey=cos(z)

        step=min((1/abs(ex + 1E-10)), (1/abs(ey + 1E-10)))

        t = 0
        while True:
            p = int(x+ex*t+0.5)
            q = int(y+ey*t+0.5)

            if(p<0 or p>=height or q<0 or q>=width):
                break

            gray=trimap[p][q]
            if(f1 is False and gray<50):
                B.append((p, q))
                f1 = True
            else:
                if(f2 is False and gray>200):
                    F.append((p, q))
                    f2 = True
                elif(f1 is True and f2 is True):
                        break
            t+=step


@njit
def prepare(height,width,image, trimap,Tu,image_info):
   for i in range(height):
        for j in range(width):
            if (trimap[i][j] != 0 and trimap[i][j] != 255):
              Tu.append((i,j))
              image_info.append((np.array([-1,-1,-1], dtype = np.int32), np.array([-1,-1,-1], dtype = np.int32), -1.0,-1.0))
            else:
              p = image[i][j]
              gray = trimap[i][j]
              if (gray == 0 ):
                  image_info.append((p,p,0.0,1.0))
              elif (gray == 255):
                  image_info.append((p,p,1.0,1.0))


@njit
def sampling(F,B,width,height,tp,trimap):
    KG = 4
    a=90
    b=29

    x = tp[0]
    y = tp[1]

    angle=(x+y)*b % a
    for step in range(KG):
        f1 = False
        f2 = False

        z=((angle + step*a)/180) * pi
        ex=sin(z)
        ey=cos(z)

        step=min((1/abs(ex + 1E-10)), (1/abs(ey + 1E-10)))

        t = 0
        while True:
            p = int(x+ex*t+0.5)
            q = int(y+ey*t+0.5)

            if(p<0 or p>=height or q<0 or q>=width):
                break

            gray=trimap[p][q]
            if(f1 is False and gray<50):
                B.append((p, q))
                f1 = True
            else:
                if(f2 is False and gray>200):
                    F.append((p, q))
                    f2 = True
                elif(f1 is True and f2 is True):
                        break
            t+=step


@njit
def gathering(image,width,height,tp,trimap):
    F = List()
    B = List()
    F.append((0,0))
    B.append((0,0))
    F = F[1:]
    B = B[1:]
    sampling(F, B, width, height, tp, trimap)

    x = tp[0]
    y = tp[1]

    pfp = pfP(x, y, F, B, image)
    gmin = 1E10

    flag = False
    tf = (0, 0) 
    tb = (0, 0) 

    for it1 in range(len(F)):
        dpf = Dp(x, y, F[it1][0], F[it1][1])

        for it2 in range(len(B)):

            gp = gP(x, y, F[it1][0], F[it1][1], B[it2][0], B[it2][1], pfp, dpf, width, height, image)
            if (gp < gmin):
                gmin = gp
                tf = (F[it1][0], F[it1][1])
                tb = (B[it2][0], B[it2][1])
                flag = True

    if (flag):
        f = image[tf[0], tf[1]]
        b = image[tb[0], tb[1]]
        sigmaf = sigma_squre(tf, height, width, image)
        sigmab = sigma_squre(tb, height, width, image)
        return (f, b, sigmaf, sigmab)


@njit
def sampleGathering(image,width,height,trimap,Tu,unknown,tau):
  holder = (np.array([-1,-1,-1], dtype = np.int32), np.array([-1,-1,-1], dtype = np.int32), -1.0,-1.0)
  for i in range(len(Tu)):
    temp = gathering(image,width,height,Tu[i],trimap)
    if(temp is not None):
      tau.append(temp)
      unknown[Tu[i][0], Tu[i][1]] = i
    else:
      tau.append(holder)
      unknown[Tu[i][0], Tu[i][1]] = i


@njit
def refine(xi,yj,height,width,image,trimap,unknown,tau):
  i1 = max(0, xi - 5)
  i2 = min(xi + 5, height - 1)
  j1 = max(0, yj - 5)
  j2 = min(yj + 5, width - 1)

  minvalue = [1E10, 1E10, 1E10]
  p = [(0,0),(0,0),(0,0)]
  num = 0
  for k in range(i1, i2 + 1):
      for l in range(j1, j2 + 1):

          temp = trimap[k][l]
          if (temp == 0 or temp == 255):
              continue

          id = unknown[k][l]
          t = tau[id]
          if (t[3] == -1.0):
              continue

          m  = mP(xi, yj, t[0], t[1], image)

          if (m > minvalue[2]):
              continue

          if (m < minvalue[0]):
              minvalue[2] = minvalue[1]
              p[2] = p[1]

              minvalue[1] = minvalue[0]
              p[1] = p[0]

              minvalue[0] = m
              p[0] = (k,l)

              num = num + 1
          elif (m < minvalue[1]):
              minvalue[2] = minvalue[1]
              p[2] = p[1]

              minvalue[1] = m
              p[1] = (k,l)

              num = num + 1
          elif (m < minvalue[2]):
              minvalue[2] = m
              p[2] = (k,l)

              num = num + 1

  num = min(num, 3)


  fb = 0
  fg = 0
  fr = 0
  bb = 0
  bg = 0
  br = 0
  sf = 0
  sb = 0

  for k in range(num):
      i  = unknown[p[k][0], p[k][1]]
      fb += tau[i][0][0]
      fg += tau[i][0][1]
      fr += tau[i][0][2]
      bb += tau[i][1][0]
      bg += tau[i][1][1]
      br += tau[i][1][2]
      sf += tau[i][2]
      sb += tau[i][3]

  fb /= (num + 1E-10)
  fg /= (num + 1E-10)
  fr /= (num + 1E-10)
  bb /= (num + 1E-10)
  bg /= (num + 1E-10)
  br /= (num + 1E-10)
  sf /= (num + 1E-10)
  sb /= (num + 1E-10)

  fc = np.array([fb, fg, fr], dtype = np.int32)
  bc = np.array([bb, bg, br], dtype = np.int32)
  pc = image[xi][yj]
  df = Dc(pc, fc)
  db = Dc(pc, bc)
  tf = fc
  tb = bc

  if (df < sf):
      fc = pc
  if (db < sb):
      bc = pc

  if (fc[0] == bc[0] and fc[1] == bc[1] and fc[2] == bc[2]):
      confidence = 1E7
  else:
      confidence = exp(-10 * mP(xi, yj, tf, tb, image))

  alphar = max(0.0, min(1.0, getalpha(pc, fc, bc)))

  return (fc, bc, alphar, confidence)


@njit
def refineSample(width, height, image, trimap, Tu, tau, unknown, image_info):
    for iter in range(len(Tu)):
      x = Tu[iter][0]
      y = Tu[iter][1]
      result = refine(x,y,height,width,image,trimap,unknown,tau)
      image_info[x * width + y] = result


@njit
def localSmooth(Tu, height, width, image, trimap, image_info, final):
    sig2 = 100.0 / (9 * pi)
    r = 3 * sqrt(sig2)
    for iter in range(len(Tu)):
        xi = Tu[iter][0]
        yj = Tu[iter][1]

        i1 = max(0, int(xi - r))
        i2 = min(int(xi + r), height - 1)
        j1 = max(0, int(yj - r))
        j2 = min(int(yj + r), width - 1)

        ptuple = image_info[xi*width + yj]

        wcfsumup = List([0.0,0.0,0.0])
        wcbsumup = List([0.0,0.0,0.0])
        wcfsumdown = 0
        wcbsumdown = 0
        wfbsumup   = 0
        wfbsundown = 0
        wasumup    = 0
        wasumdown  = 0

        for k in range(i1, i2 + 1):
            for l in range(j1, j2 + 1):
                qtuple = image_info[k*width + l]

                d = Dp(xi, yj, k, l)

                if (d > r):
                    continue

                if (d == 0):
                    wc = exp(-(d * d) / sig2) * qtuple[3]
                else:
                    wc = exp(-(d * d) / sig2) * qtuple[3] * abs(qtuple[2] - ptuple[2])

                wcfsumdown += wc * qtuple[2]
                wcbsumdown += wc * (1 - qtuple[2])

                wcfsumup[0] += wc * qtuple[2] * qtuple[0][0]
                wcfsumup[1] += wc * qtuple[2] * qtuple[0][1]
                wcfsumup[2] += wc * qtuple[2] * qtuple[0][2]

                wcbsumup[0] += wc * (1 - qtuple[2]) * qtuple[1][0]
                wcbsumup[1] += wc * (1 - qtuple[2]) * qtuple[1][1]
                wcbsumup[2] += wc * (1 - qtuple[2]) * qtuple[1][2]


                wfb = qtuple[3] * qtuple[2] * (1 - qtuple[2])
                wfbsundown += wfb
                wfbsumup   += wfb * sqrt(Dc(qtuple[0], qtuple[1]))

                delta = 0
                if (trimap[k][l] == 0 or trimap[k][l] == 255):
                    delta = 1
                wa = qtuple[3] * exp(-(d * d) / sig2) + delta
                wasumdown += wa
                wasumup   += wa * qtuple[2]

        cp = image[xi][yj]

        fp = (0,0,0)
        bp = (0,0,0)

        bp = (min(255.0, max(0.0,wcbsumup[0] / (wcbsumdown + 1E-10))),
              min(255.0, max(0.0,wcbsumup[1] / (wcbsumdown + 1E-10))),
              min(255.0, max(0.0,wcbsumup[2] / (wcbsumdown + 1E-10))))
        
        fp = (min(255.0, max(0.0,wcfsumup[0] / (wcfsumdown + 1E-10))),
              min(255.0, max(0.0,wcfsumup[1] / (wcfsumdown + 1E-10))),
              min(255.0, max(0.0,wcfsumup[2] / (wcfsumdown + 1E-10))))

        dfb  = wfbsumup / (wfbsundown + 1E-10)

        conp = min(1.0, sqrt(Dc(fp, bp)) / (dfb + 1E-10)) * exp(-10 * mP(xi, yj, fp, bp, image))
        alp  = wasumup / (wasumdown + 1E-10)

        alpha_t = conp * getalpha(cp, fp, bp) + (1 - conp) * max(0.0, min(alp, 1.0))
        final[xi][yj] = alpha_t*255


def main():
  image_path  = str(input('give a path (png format required):'))

  start = time.time()
  trimap_path = image_path[:len(image_path)-4] + '_trimap.png'
  alpha_path = image_path[:len(image_path)-4] + '_alpha.png'

  trimap =  cv.imread(trimap_path,0).astype(np.int32)
  width = trimap.shape[1]
  height = trimap.shape[0]

  image = cv.imread(image_path)
  image = cv.resize(image, (width,height)).astype(np.int32)

  threadsperblock = (16,16)
  blockspergrid_x = int(ceil(height/threadsperblock[0]))
  blockspergrid_y = int(ceil(width/threadsperblock[1]))
  blockspergrid = (blockspergrid_x,blockspergrid_y)


  dImg = cuda.to_device(image)
  dTri = cuda.to_device(trimap)
  dN = cuda.to_device(trimap)

  cuda.synchronize()
  expansion[blockspergrid,threadsperblock](dImg,dTri,dN,height,width)
  cuda.synchronize()

  dout = dN.copy_to_host().astype(np.uint8)
  final = dout.copy()

  Tu = List()
  Tu.append((0,0))
  Tu = Tu[1:]
  image_info = List()
  image_info.append((np.array([-1,-1,-1], dtype = np.int32), np.array([-1,-1,-1], dtype = np.int32), -1.0,-1.0))
  image_info = image_info[1:]
  prepare(height,width,image,dout,Tu,image_info)

  tau = List()
  tau.append((np.array([-1,-1,-1], dtype = np.int32), np.array([-1,-1,-1], dtype = np.int32), -1.0,-1.0))
  tau = tau[1:]
  unknown = np.zeros((height,width),  dtype = int)
  sampleGathering(image,width,height,dout,Tu,unknown,tau)

  refineSample(width, height, image, dout, Tu, tau, unknown, image_info)
  # for i in range(height):
  #       for j in range(width):
  #         final[i][j] = image_info[i*width+j][2]*255

  localSmooth(Tu, height, width, image, dout, image_info, final)

  cv.imwrite(alpha_path, final)

  end = time.time()
  print(end - start)

if __name__ == "__main__":
    main()