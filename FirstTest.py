import cv2
import numpy as np

imgTarget = cv2.imread("ArUco robo.jpg")

if imgTarget is None:
    print("Erro ao carregar a imagem alvo.")
    exit()

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Erro ao abrir webcam.")
    exit()

grayTarget = cv2.cvtColor(imgTarget, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=1500)
keyPoints1, descript1 = orb.detectAndCompute(grayTarget, None)

if descript1 is None:
    print("Erro: a imagem alvo não gerou descritores. Escolha uma imagem com mais detalhes.")
    exit()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

found_count = 0
lost_count = 0
detected = False
last_corners = None

while True:
    ok, imgWebCam = cam.read()
    if not ok:
        break

    imgResult = imgWebCam.copy()
    grayWebCam = cv2.cvtColor(imgWebCam, cv2.COLOR_BGR2GRAY)

    keyPoints2, descript2 = orb.detectAndCompute(grayWebCam, None)

    encontrou_agora = False
    good_count = 0
    inliers = 0
    corners = None

    if descript2 is not None and len(keyPoints2) > 0:
        matches = bf.knnMatch(descript1, descript2, k=2)

        goodmatches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.8 * n.distance:   # mais tolerante
                    goodmatches.append(m)

        good_count = len(goodmatches)

        if good_count >= 8:   # antes estava muito alto
            src_pts = np.float32([keyPoints1[m.queryIdx].pt for m in goodmatches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keyPoints2[m.trainIdx].pt for m in goodmatches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None and mask is not None:
                inliers = int(mask.sum())

                if inliers >= 6:   # mais tolerante
                    encontrou_agora = True

                    h, w = grayTarget.shape
                    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    corners = cv2.perspectiveTransform(pts, H)
                    last_corners = corners

    if encontrou_agora:
        found_count += 1
        lost_count = 0
    else:
        lost_count += 1
        found_count = 0

    if found_count >= 2:   # confirma mais rápido
        detected = True

    if lost_count >= 8:    # demora mais para “desligar”
        detected = False
        last_corners = None

    if detected:
        cv2.putText(imgResult, "Padrao encontrado", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if last_corners is not None:
            cv2.polylines(imgResult, [np.int32(last_corners)], True, (255, 0, 0), 3)
    else:
        cv2.putText(imgResult, "Padrao nao encontrado", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(imgResult, f"Good matches: {good_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(imgResult, f"Inliers: {inliers}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Deteccao", imgResult)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()