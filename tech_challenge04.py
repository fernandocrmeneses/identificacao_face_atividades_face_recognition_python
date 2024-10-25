import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
import face_recognition
import mediapipe as mp


class TechChallenge04:
    
    def __init__(self):
        self.width = None
        self.height = None
        self.fps = None
        self.total_frames = None
        self.emotions_translation = {
            "angry": "raiva",
            "disgust": "desgosto",
            "fear": "medo",
            "happy": "feliz",
            "sad": "triste",
            "surprise": "surpresa",
            "neutral": "neutro"
    }
        
    def extrair_frames_do_video(self, video_path, image_folder):
        # Captura de vídeo
        cap = cv2.VideoCapture(video_path)
        self.obtem_propriedades_video(video_path)
 
        if not self.video_foi_aberto_corretamente(cap):
            return

        frame_count = 0
        for _ in tqdm(range(self.total_frames), desc="Extraindo frames do vídeo"):
            # Ler cada frame do vídeo
            ret, frame = cap.read()
        
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Nome do arquivo para cada frame (exemplo: frame_001.jpg)
            image_filename = f"desconhecido_{frame_count:04d}.jpg"
            image_path = os.path.join(image_folder, image_filename)

            # Salvar o frame como imagem
            cv2.imwrite(image_path, gray_frame)
            frame_count += 1

        cap.release()
        
    def carrega_imagens_pasta(self, folder):
        conhecido_face_encodings = []
        conhecido_nome_faces = []
        
        image_files = [f for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".png")]
        # Percorrer todos os arquivos na pasta fornecida
        for nome_arquivo in tqdm(image_files, desc="Carregando imagens e decodificacoes"):
            #if (image_files.index(nome_arquivo) <= 10):
                
                image_path = os.path.join(folder, nome_arquivo)
                image = face_recognition.load_image_file(image_path)
                # Obter as codificações faciais (assumindo uma face por imagem)
                face_encodings = face_recognition.face_encodings(image)
            
                if face_encodings:
                    face_encoding = face_encodings[0]
                    # Extrair o nome do arquivo, removendo o sufixo numérico e a extensão
                    name = os.path.splitext(nome_arquivo)[0][:-1]
                    # Adicionar a codificação e o nome às listas
                    conhecido_face_encodings.append(face_encoding)
                    conhecido_nome_faces.append(name)
                    
        return conhecido_face_encodings, conhecido_nome_faces
    
    def obtem_propriedades_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
    def video_foi_aberto_corretamente(self, cap):
        if not cap.isOpened():
            print("Erro ao abrir o vídeo.")
            return False
        return True
        
    def define_codec_cria_obj_video_writer(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
        return cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
    
    def traduz_emocao_para_portugues(self, emocao_dominante):
        return self.emotions_translation.get(emocao_dominante, emocao_dominante)
    
    def verifica_canal_cores(self, frame):
        if len(frame.shape) == 2:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame
    
    def detecta_poses(self, video_path, output_path):
        # Inicializar o MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mp_drawing = mp.solutions.drawing_utils

        # Capturar vídeo do arquivo especificado
        cap = cv2.VideoCapture(video_path)

        if not self.video_foi_aberto_corretamente(cap):
            return
            
        self.obtem_propriedades_video(video_path)
        out = self.define_codec_cria_obj_video_writer(output_path)

        # Loop para processar cada frame do vídeo com barra de progresso
        for _ in tqdm(range(self.total_frames), desc="Processando vídeo"):
            # Ler um frame do vídeo
            ret, frame = cap.read()

            # Se não conseguiu ler o frame (final do vídeo), sair do loop
            if not ret:
                break

            rgb_frame = self.verifica_canal_cores(frame)
            results = pose.process(rgb_frame)

            # Desenhar as anotações da pose no frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Escrever o frame processado no vídeo de saída
            out.write(frame)

            # Exibir o frame processado
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Liberar a captura de vídeo e fechar todas as janelas
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    def detecta_atividades(self, frame, rgb_frame):
        # Inicializar o MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mp_drawing = mp.solutions.drawing_utils
        # Processar o frame para detectar a pose
        resultado = pose.process(rgb_frame)
        # Desenhar as anotações da pose no frame
        if resultado.pose_landmarks:
            mp_drawing.draw_landmarks(frame, resultado.pose_landmarks, mp_pose.POSE_CONNECTIONS)
         
         
    def detecta_faces_emocoes(self,video_path, output_path, face_encodings_conhecido, nomes_faces_conhecidas):
        # Capturar vídeo do arquivo especificado
        cap = cv2.VideoCapture(video_path)

        if not self.video_foi_aberto_corretamente(cap): 
            return
        
        self.obtem_propriedades_video(video_path)

        out = self.define_codec_cria_obj_video_writer(output_path)

        # Loop para processar cada frame do vídeo com barra de progresso
        for _ in tqdm(range(self.total_frames), desc="Processando vídeo"):
            # Ler um frame do vídeo
            ret, frame = cap.read()

            # Se não conseguiu ler o frame (final do vídeo), sair do loop
            if not ret:
                break

            rgb_frame = self.verifica_canal_cores(frame)
            
            # Analisar o frame para detectar faces e expressões
            resultado = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')
           
            # Obter as localizações e codificações das faces conhecidas no frame
            localizacao_face = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, localizacao_face)
            # Inicializar uma lista de nomes para as faces detectadas
            lista_nomes_faces = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(face_encodings_conhecido, face_encoding, tolerance=0.6)
                name = "Desconhecido"
                face_distances = face_recognition.face_distance(face_encodings_conhecido, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = nomes_faces_conhecidas[best_match_index]
                lista_nomes_faces.append(name)

            # Iterar sobre cada face detectada pelo DeepFace
            for face in resultado:
                # Obter a caixa delimitadora da face
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            
                # Obter a emoção dominante
                emocao_dominante = face['dominant_emotion']
                emocao_dominante_traduzida = self.traduz_emocao_para_portugues(emocao_dominante)
                # Desenhar um retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Escrever a emoção dominante acima da face
                cv2.putText(frame, emocao_dominante_traduzida, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Associar a face detectada pelo DeepFace com as faces conhecidas
                for (top, direita, abaixo, esquerda), nome in zip(localizacao_face, lista_nomes_faces):
                    if x <= esquerda <= x + w and y <= top <= y + h:
                        # Escrever o nome abaixo da face
                        cv2.putText(frame, nome, (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        break
                    
            
             # Escrever o frame processado no vídeo de saída
            out.write(frame)

            # Exibir o frame processado
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                          
            # Escrever o frame processado no vídeo de saída
            out.write(frame)
        # Liberar a captura de vídeo e fechar todas as janelas
        cap.release()
        out.release()
        #cv2.destroyAllWindows()


tech = TechChallenge04()
# Caminho para o arquivo de vídeo na mesma pasta do script
diretorio_script = os.path.dirname(os.path.abspath(__file__))

input_video_path = os.path.join(diretorio_script, r'video\video_tech04.mp4')  
output_video_path = os.path.join(diretorio_script, r'output\output_video.mp4')
poses_video_path = os.path.join(diretorio_script,r'output\poses\output_video_03.mp4') 

# Caminho para a pasta de imagens com rostos conhecidos
image_folder = "imagens"
tech.extrair_frames_do_video(input_video_path,os.path.join(diretorio_script,image_folder))

# Carregar imagens e codificações
conhecido_face_encondings, conhecido_nome_faces = tech.carrega_imagens_pasta(image_folder)
    
tech.detecta_faces_emocoes(input_video_path, output_video_path, conhecido_face_encondings, conhecido_nome_faces)

tech.detecta_poses(poses_video_path, output_video_path)