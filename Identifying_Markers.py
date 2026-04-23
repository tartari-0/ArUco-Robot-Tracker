import cv2
import cv2.aruco as aruco

def main():
    # Inicializa a câmera. 
    # '0' geralmente é a webcam do notebook. '1' ou '2' pode ser sua câmera USB.
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    # 1. DEFINIÇÃO DO DICIONÁRIO (MUITO IMPORTANTE)
    # Você deve alterar DICT_4X4_50 para a família exata de marcadores que você imprimiu.
    # Ex: se seus PDFs são 6x6, mude para cv2.aruco.DICT_6X6_250.
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # 2. CONFIGURAÇÃO DO DETECTOR
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    print("Pressione 'q' para sair do programa.")

    while True:
        # Lê um frame da câmera
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o frame.")
            break

        # Converte a imagem para escala de cinza (o algoritmo funciona melhor assim)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta os marcadores na imagem
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        # Se encontrar algum marcador (ids não for None)
        if ids is not None:
            # Desenha o contorno e o ID diretamente sobre o frame colorido
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Printa no terminal para você saber que está lendo
            # O flatten() transforma a matriz em uma lista simples
            print(f"IDs detectados na tela: {ids.flatten()}")

        # Mostra o resultado em tempo real em uma janela
        cv2.imshow('Camera - Deteccao ArUco', frame)

        # Aguarda 1 milissegundo e verifica se a tecla 'q' foi pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos do PC ao finalizar
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()