import cv2
import cv2.aruco as aruco
from collections import deque
import numpy as np

def calculate_marker_center(corners):
    # Calcula o centro geométrico tirando a média dos 4 cantos (x, y)
    center = np.mean(corners[0], axis=0).astype(int)
    return tuple(center)

def main():
    # Inicializa a câmera.
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    # 1. CONFIGURAÇÃO ARUCO (Mantenha o dicionário do código anterior)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    # 2. DEFINIÇÃO DOS IDs DOS MARCADORES (Conforme sua organização)
    ROBOT_ID = 0
    # Arena: (2=inf-esq, 4=sup-esq, 8=inf-dir, 5=sup-dir)
    ARENA_IDS = [2, 4, 8, 5]

    # 3. HISTÓRICO DA TRAJETÓRIA (Armazena os últimos 500 pontos)
    trajectory_history = deque(maxlen=500)

    print("Pressione 'q' para sair e 'c' para limpar a trajetória.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Captura as dimensões para a janela do mapa
        h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        # ----------------------------------------------------
        # JANELA 1: FEED ORIGINAL COM DETECÇÕES
        # ----------------------------------------------------
        original_frame = frame.copy()
        if ids is not None:
            aruco.drawDetectedMarkers(original_frame, corners, ids)
        cv2.imshow('1. Câmera - Detecção ArUco', original_frame)

        # ----------------------------------------------------
        # JANELA 2: REPRESENTAÇÃO E MAPA DE TRAJETÓRIA
        # ----------------------------------------------------
        # Cria uma tela branca do mesmo tamanho da câmera
        trajectory_map = np.ones((h, w, 3), dtype=np.uint8) * 255
        arena_corners = {}
        robot_center_now = None

        if ids is not None:
            # Processa cada marcador encontrado
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                center = calculate_marker_center(marker_corners)
                
                # É o robô?
                if marker_id == ROBOT_ID:
                    robot_center_now = center
                # É um canto da arena?
                elif marker_id in ARENA_IDS:
                    arena_corners[marker_id] = center

            # --- A. DESENHAR A ARENA (Se todos os cantos foram vistos) ---
            # Verificação se todos os 4 cantos estão na tela
            if len(arena_corners) == 4:
                # Constrói o polígono na ordem: sup-esq -> sup-dir -> inf-dir -> inf-esq
                pts = np.array([
                    arena_corners[4], # Canto sup-esq
                    arena_corners[5], # Canto sup-dir
                    arena_corners[8], # Canto inf-dir
                    arena_corners[2]  # Canto inf-esq
                ], np.int32).reshape((-1, 1, 2))
                
                # Desenha o polígono pontilhado da arena
                # (Representação diagramática)
                cv2.polylines(trajectory_map, [pts], True, (100, 100, 100), 1, lineType=cv2.LINE_AA)
            else:
                cv2.putText(trajectory_map, "Aguardando Arena (4 cantos)...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

            # --- B. PROCESSAR E DESENHAR O ROBÔ ---
            if robot_center_now is not None:
                # Adiciona a posição atual ao histórico
                trajectory_history.append(robot_center_now)
                
                # Desenha o robô como um pequeno retângulo (id0)
                robot_size = 30
                rx, ry = robot_center_now
                cv2.rectangle(trajectory_map, (rx - robot_size//2, ry - robot_size//2),
                              (rx + robot_size//2, ry + robot_size//2), (0, 0, 0), -1)

            # --- C. DESENHAR A TRAJETÓRIA COMPLETA ---
            if len(trajectory_history) >= 2:
                # Desenha as linhas que conectam os pontos históricos
                for i in range(1, len(trajectory_history)):
                    cv2.line(trajectory_map, trajectory_history[i - 1], 
                             trajectory_history[i], (200, 50, 50), 2, cv2.LINE_AA)

        else:
            cv2.putText(trajectory_map, "Nenhum ArUco detectado.", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Mostra a janela do mapa
        cv2.imshow('2. Mapa de Trajetória', trajectory_map)

        # 4. ENTRADAS DO USUÁRIO
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Tecla 'c' limpa a trajetória
            trajectory_history.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()