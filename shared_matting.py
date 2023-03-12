import cv2 as cv
from math import sqrt
from math import ceil
from math import cos
from math import sin
from math import pi
from math import exp
import numpy as np
import time
from numba import cuda

@cuda.jit
def Dp(x_1,y_1,x_2,y_2):
  return sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)

@cuda.jit
def Dc(tuple1, tuple2):
  return (tuple1[0]-tuple2[0])**2 + (tuple1[1]-tuple2[1])**2 + (tuple1[2]-tuple2[2])**2

@cuda.jit
def pfP(x, y, f_0, f_1, b_0, b_1, image):
  fmin = 1E10
  for i in range(len(f_0)):
    if(f_0[i] == -1 or f_1[i] == -1):
      continue
    else:
      fp = eP(x, y, f_0[i], f_1[i], image)
      if (fp < fmin):
        fmin = fp
  
  bmin = 1E10
  for i in range(len(b_0)):
    if(b_0[i] == -1 or b_1[i] == -1):
      continue
    else:
      bp = eP(x, y, b_0[i], b_1[i], image)
      if (bp < bmin):
        bmin = bp
      
  return bmin / (fmin + bmin + 1E-10)

@cuda.jit
def getalpha(p, f, b):
  alpha = (((p[0] - b[0]) * (f[0] - b[0]) + 
            (p[1] - b[1]) * (f[1] - b[1]) + 
            (p[2] - b[2]) * (f[2] - b[2]))
          /((f[0] - b[0]) ** 2 + 
            (f[1] - b[1]) ** 2 + 
            (f[2] - b[2]) ** 2 + 1E-6))
  return min(1, max(0, alpha))

@cuda.jit
def mP(i, j, f, b, image):
  p = image[i][j]
  alpha = getalpha(p, f, b)

  result = sqrt((p[0] - alpha * f[0] - (1 - alpha) * b[0]) * (p[0] - alpha * f[0] - (1 - alpha) * b[0]) +
                  (p[1] - alpha * f[1] - (1 - alpha) * b[1]) * (p[1] - alpha * f[1] - (1 - alpha) * b[1]) +
                  (p[2] - alpha * f[2] - (1 - alpha) * b[2]) * (p[2] - alpha * f[2] - (1 - alpha) * b[2]))
  return result / 255

@cuda.jit
def aP(x, y, pf, f, b, image):
  p = image[x][y]
  alpha = getalpha(p, f, b)

  return pf + (1 - 2 * pf) * alpha

@cuda.jit
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

@cuda.jit
def gP(x, y, f_x, f_y, b_x, b_y, pf, dpf, height, width, image):
  f = image[f_x, f_y]
  b = image[b_x, b_y]
  dpb = Dp(x, y, b_x, b_y)

  tn = nP(x, y, f, b, height, width, image) 
  ta = aP(x, y, pf, f, b ,image) ** 2

  return (tn ** 3) * (ta ** 2) * dpf * (dpb ** 4)

@cuda.jit
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
                  dis = Dp(tx, ty, l, l1)

                  if (dis > KI):
                      continue

                  distanceColor = Dc(image[tx][ty], image[l][l1])
                  if (distanceColor <= KC):
                    flag = True
                    new_trimap[tx][ty] = gray
                    
                if (flag):
                  break

                gray = trimap[l][l2]
                if (gray == 0 or gray == 255):
                  dis = Dp(tx, ty, l, l2)

                  if (dis > KI):
                    continue

                  distanceColor = Dc(image[tx][ty], image[l][l2])
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
                  dis = Dp(tx, ty, k2, l)

                  if (dis > KI):
                    continue

                  distanceColor = Dc(image[tx][ty], image[k1][l])
                  if (distanceColor <= KC):
                    flag = True
                    new_trimap[tx][ty] = gray
                    
                if (flag):
                    break

                gray = trimap[k2][l]
                if (gray == 0 or gray == 255):
                    dis = Dp(tx, ty, k1, l)

                    if (dis > KI):
                        continue

                    distanceColor = Dc(image[tx][ty], image[k2][l])
                    if (distanceColor <= KC):
                        flag = True
                        new_trimap[tx][ty] = gray
                        
                if (flag):
                    break
              else:
                  break
          else:
            break
          
@cuda.jit
def sigma_squre(x, y, height, width, image):
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
def sampleGathering(image,width,height,trimap,tau0,tau1,tau2,tau3):
  tx = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
  ty = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y

  if tx < height and ty < width:
    if (trimap[tx][ty] != 0 and trimap[tx][ty] != 255):
      KG = 4
      a=90
      b=29

      F_0 = cuda.local.array(shape=4,dtype=np.int32)
      F_1 = cuda.local.array(shape=4,dtype=np.int32)
      B_0 = cuda.local.array(shape=4,dtype=np.int32)
      B_1 = cuda.local.array(shape=4,dtype=np.int32)
     
      for init in range(4):
        F_0[init] = -1
        F_1[init] = -1
        B_0[init] = -1
        B_1[init] = -1

      angle=(tx+ty)*b % a
      for i in range(KG):
        f1 = False
        f2 = False

        tz=((angle + i*a)/180) * pi
        ex=sin(tz)
        ey=cos(tz)

        step=min((1/abs(ex + 1E-10)), (1/abs(ey + 1E-10)))

        t = 0
        while True:
          p = int(tx+ex*t+0.5)
          q = int(ty+ey*t+0.5)

          if(p<0 or p>=height or q<0 or q>=width):
              break

          gray=trimap[p][q] 
          if(f1 is False and gray<50):
            B_0[i] = p
            B_1[i] = q
            f1 = True
          else:
            if(f2 is False and gray>200):
              F_0[i] = p
              F_1[i] = q
              f2 = True
            elif(f1 is True and f2 is True):
                    break
          t+=step

      pfp = pfP(tx, ty, F_0, F_1, B_0, B_1, image)

      gmin = 1E10
      flag = False
      tf_0 = 0 
      tf_1 = 0
      tb_0 = 0
      tb_1 = 0

      for it1 in range(len(F_0)):
        if(F_0[it1] == -1 or F_1[it1] == -1):
          continue
        else:
          dpf = Dp(tx, ty, F_0[it1], F_1[it1])

          for it2 in range(len(B_0)):
            if(B_0[it2] == -1 or B_1[it2] == -1):
              continue
            else:
              gp = gP(tx, ty, F_0[it1], F_1[it1], B_0[it2], B_1[it2], pfp, dpf, width, height, image)
              if (gp < gmin):
                  gmin = gp
                  tf_0 = F_0[it1]
                  tf_1 = F_1[it1]
                  tb_0 = B_0[it2]
                  tb_1 = B_1[it2]
                  flag = True

      if (flag):
          f = (image[tf_0][tf_1][0], image[tf_0][tf_1][1], image[tf_0][tf_1][2])
          b = (image[tb_0][tb_1][0], image[tb_0][tb_1][1], image[tb_0][tb_1][2])
          sigmaf = sigma_squre(tf_0, tf_1, height, width, image)
          sigmab = sigma_squre(tb_0, tb_1, height, width, image)
          tau0[tx][ty] = f
          tau1[tx][ty] = b
          tau2[tx][ty] = sigmaf
          tau3[tx][ty] = sigmab
      else:
          tau0[tx][ty] = (-1.0,-1.0,-1.0)
          tau1[tx][ty] = (-1.0,-1.0,-1.0)
          tau2[tx][ty] = -1.0
          tau3[tx][ty] = -1.0
    else:
          tau0[tx][ty] = (-1.0,-1.0,-1.0)
          tau1[tx][ty] = (-1.0,-1.0,-1.0)
          tau2[tx][ty] = -1.0
          tau3[tx][ty] = -1.0

@cuda.jit
def refineSample(image, width, height, trimap, tau0, tau1, tau2, tau3, info0, info1, info2, info3):
  tx = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
  ty = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y

  if tx < height and ty < width:
    if (trimap[tx][ty] != 0 and trimap[tx][ty] != 255):
        i1 = max(0, tx - 5)
        i2 = min(tx + 5, height - 1)
        j1 = max(0, ty - 5)
        j2 = min(ty + 5, width - 1)

        minvalue = cuda.local.array(shape=3,dtype=np.float64)
        p_0 = cuda.local.array(shape=3,dtype=np.int32)
        p_1 = cuda.local.array(shape=3,dtype=np.int32)

        for init in range(3):
          minvalue[init] = 1E10
          p_0[init] = 0
          p_1[init] = 0

        num = 0

        for k in range(i1, i2 + 1):
          for l in range(j1, j2 + 1):

            temp = trimap[k][l];
            if (temp == 0 or temp == 255):
              continue

            if (tau2[k][l] == -1.0):
              continue

            m  = mP(tx, ty, tau0[k][l], tau1[k][l], image)

            if (m > minvalue[2]):
              continue

            if (m < minvalue[0]):
              minvalue[2] = minvalue[1]
              p_0[2] = p_0[1]
              p_1[2] = p_1[1]

              minvalue[1] = minvalue[0]
              p_0[1] = p_0[0]
              p_1[1] = p_1[0]

              minvalue[0] = m
              p_0[0] = k
              p_1[0] = l

              num = num + 1
            elif (m < minvalue[1]):
              minvalue[2] = minvalue[1]
              p_0[2] = p_0[1]
              p_1[2] = p_1[1]

              minvalue[1] = m
              p_0[1] = k
              p_1[1] = l

              num = num + 1
            elif (m < minvalue[2]):
              minvalue[2] = m
              p_0[2] = k
              p_1[2] = l

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
          fb += tau0[p_0[k]][p_1[k]][0]
          fg += tau0[p_0[k]][p_1[k]][1]
          fr += tau0[p_0[k]][p_1[k]][2]
          bb += tau1[p_0[k]][p_1[k]][0]
          bg += tau1[p_0[k]][p_1[k]][1]
          br += tau1[p_0[k]][p_1[k]][2]
          sf += tau2[p_0[k]][p_1[k]]
          sb += tau3[p_0[k]][p_1[k]]

        fb /= (num + 1E-10)
        fg /= (num + 1E-10)
        fr /= (num + 1E-10)
        bb /= (num + 1E-10)
        bg /= (num + 1E-10)
        br /= (num + 1E-10)
        sf /= (num + 1E-10)
        sb /= (num + 1E-10)

        fc = (fb, fg, fr)
        bc = (bb, bg, br)
        pc = image[tx][ty]
        df = Dc(pc, fc)
        db = Dc(pc, bc)
        tf = fc
        tb = bc

        if (df < sf):
          fc = (pc[0], pc[1], pc[2])
        if (db < sb):
          bc = (pc[0], pc[1], pc[2])

        if (fc[0] == bc[0] and fc[1] == bc[1] and fc[2] == bc[2]):
          confidence = 1E7
        else:
          confidence = exp(-10 * mP(tx, ty, tf, tb, image))

        alphar = max(0.0, min(1.0, getalpha(pc, fc, bc)))


        info0[tx][ty] = fc
        info1[tx][ty] = bc
        info2[tx][ty] = alphar
        info3[tx][ty] = confidence

    else:
      c = (image[tx][ty][0], image[tx][ty][1], image[tx][ty][2])
      if (trimap[tx][ty] == 0 ):
        info0[tx][ty] = c
        info1[tx][ty] = c
        info2[tx][ty] = 0.0
        info3[tx][ty] = 1.0
      elif (trimap[tx][ty] == 255):
        info0[tx][ty] = c
        info1[tx][ty] = c
        info2[tx][ty] = 1.0
        info3[tx][ty] = 1.0  

@cuda.jit
def localSmooth(image, width, height, trimap, info0, info1, info2, info3, final):
  sig2 = 100.0 / (9 * pi)
  r = 3 * sqrt(sig2)

  tx = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
  ty = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y

  if tx < height and ty < width:
    if (trimap[tx][ty] != 0 and trimap[tx][ty] != 255):
      i1 = max(0, int(tx - r));
      i2 = min(int(tx + r), height - 1)
      j1 = max(0, int(ty - r))
      j2 = min(int(ty + r), width - 1)

      ptuple = (info0[tx][ty], info1[tx][ty], info2[tx][ty], info3[tx][ty])

      wcfsumup = cuda.local.array(shape=3,dtype=np.float32)
      wcbsumup = cuda.local.array(shape=3,dtype=np.float32)

      for init in range(3):
        wcfsumup[init] = 0.0
        wcbsumup[init] = 0.0

      wcfsumdown = 0
      wcbsumdown = 0
      wfbsumup   = 0
      wfbsundown = 0
      wasumup    = 0
      wasumdown  = 0

      for k in range(i1, i2 + 1):
        for l in range(j1, j2 + 1):
          qtuple = (info0[k][l], info1[k][l], info2[k][l], info3[k][l])

          d = Dp(tx, ty, k, l)

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

      cp = image[tx][ty]

      bp = (min(255.0, max(0.0,wcbsumup[0] / (wcbsumdown + 1E-10))),
            min(255.0, max(0.0,wcbsumup[1] / (wcbsumdown + 1E-10))),
            min(255.0, max(0.0,wcbsumup[2] / (wcbsumdown + 1E-10))))
      
      fp = (min(255.0, max(0.0,wcfsumup[0] / (wcfsumdown + 1E-10))),
            min(255.0, max(0.0,wcfsumup[1] / (wcfsumdown + 1E-10))),
            min(255.0, max(0.0,wcfsumup[2] / (wcfsumdown + 1E-10))))

      dfb  = wfbsumup / (wfbsundown + 1E-10)

      conp = min(1.0, sqrt(Dc(fp, bp)) / (dfb + 1E-10)) * exp(-10 * mP(tx, ty, fp, bp, image))
      alp  = wasumup / (wasumdown + 1E-10)

      alpha_t = conp * getalpha(cp, fp, bp) + (1 - conp) * max(0.0, min(alp, 1.0))
      final[tx][ty] = alpha_t*255

def main():
  image_path  = str(input('give a path (png format required):'))

  start = time.time()
  trimap_path = image_path[:len(image_path)-4] + '_trimap.png'
  alpha_path = image_path[:len(image_path)-4] + '_alpha.png'
  result_path = image_path[:len(image_path)-4] + '_result.png'

  trimap =  cv.imread(trimap_path,0).astype(np.int32)
  width = trimap.shape[1]
  height = trimap.shape[0]

  image = cv.imread(image_path)
  image = cv.resize(image, (width,height)).astype(np.float32)

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

  tau0 = cuda.to_device(image)
  tau1 = cuda.to_device(image)
  tau2 = cuda.to_device(np.zeros((height,width), dtype = np.float32))
  tau3 = cuda.to_device(np.zeros((height,width), dtype = np.float32))

  cuda.synchronize()
  sampleGathering[blockspergrid,threadsperblock](dImg,width,height,dN,tau0,tau1,tau2,tau3)
  cuda.synchronize()

  info0 = cuda.to_device(image)
  info1 = cuda.to_device(image)
  info2 = cuda.to_device(np.zeros((height,width), dtype = np.float32))
  info3 = cuda.to_device(np.zeros((height,width), dtype = np.float32))

  cuda.synchronize()
  refineSample[blockspergrid,threadsperblock](dImg,width,height,dN,tau0,tau1,tau2,tau3,info0,info1,info2,info3)
  cuda.synchronize()

  dout = dN.copy_to_host().astype(np.uint8)
  final = cuda.to_device(dout.copy())

  cuda.synchronize()
  localSmooth[blockspergrid,threadsperblock](dImg, width, height, dN, info0, info1, info2, info3, final)
  cuda.synchronize()

  result = final.copy_to_host()
  cv.imwrite(result_path, result)

  end = time.time()
  print(end - start)

if __name__ == '__main__': 
  main()