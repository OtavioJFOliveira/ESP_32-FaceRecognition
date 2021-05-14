from mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray


detector = MTCNN() # Usado para detectar rostos em uma foto

def extrair_face(arquivo, size=(160,160)): # Passando o arquivo e definindo o tamanho no qual a imagem recortada deve ter

    img = Image.open(arquivo) # Caminho completo do arquivo
    img = img.convert('RGB') # Convertendo a imagem em RGB para todas ficarem no mesmo padrão
    array = asarray(img) # Converte o objeto que está em PILLOW para NUMPY
    results = detector.detect_faces(array) # Results armazena uma especie de Json de resultados, em relação as posições do rosto e outras caractersiticas
    x1, y1, width, height = results[0]['box'] # Extraindo os pontos x1,y1 e as dimensoes do rosto
    x2, y2 = x1 + width, y1 + height

    face = array[y1:y2,x1:x2] # Comando NUMPY para extrair um recorte do rosto apenas

    image = Image.fromarray(face) # Atribui a imagem o array do NUMPY
    image = image.resize(size) # Redefine as dimensões da imagem para deixar quadrado, se não passar o tamanho deixa 160x160 por padrão

    return image

########################################################################################
def flip_image(image):
    img = image.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def rotate_image90(image):
    img90 = image.transpose(Image.ROTATE_90)
    return img90

def rotate_image270(image):
    img270 = image.transpose(Image.ROTATE_270)
    return img270
def rotate_image180(image):
    img180 = image.transpose(Image.ROTATE_180)
    return img180

def load_fotos(directory_scr, directory_target):

    for filename in listdir(directory_scr): #Listar todas as fotos de cada diretorio

        path         = directory_scr + filename
        path_tg      = directory_target + filename
        path_tg_flip = directory_target + "flip-" + filename

        path90 = directory_target + "90-" + filename
        path180 = directory_target + "180-" + filename
        path270 = directory_target + "270-" + filename

        try:
            RFace = extrair_face(path)
            F90 = rotate_image90(RFace)
            F180 = rotate_image180(RFace)
            F270 = rotate_image270(RFace)

            F90.save(path90, "JPEG", quality=100, optimize=True, progressive=True)
            F180.save(path180, "JPEG", quality=100, optimize=True, progressive=True)
            F270.save(path270, "JPEG", quality=100, optimize=True, progressive=True)
            #face = extrair_face(path)
            #flip = flip_image(face)

            #face.save(path_tg, "JPEG", quality=100, optimize=True, progressive=True) # Salva a imagem redimensionada na pasta FACES
            #flip.save(path_tg_flip, "JPEG", quality=100, optimize=True, progressive=True) # Salva a imagem redimensionada na pasta FACES

        except:
            print("Erro na imagem: {}".format(path))


def load_dir(directory_src, directory_target):

    for subdir in listdir(directory_src):

        path = directory_src + subdir + "\\"

        path_tg = directory_target + subdir + "\\"

        if not isdir(path):
            continue

        load_fotos(path, path_tg)

if __name__ == '__main__':

    load_dir("C:\\Users\\otavi\\Desktop\\Mestrado\\Imagens\\Fotos\\",
             "C:\\Users\\otavi\\Desktop\\Mestrado\\Imagens\\Faces\\")