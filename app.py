import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
import shutil
import glob
from sklearn.model_selection import train_test_split
from ultralytics import RTDETR
from ultralytics import YOLO

Path=r'C:\Users\USER\Desktop\LAO\Image_Recognition\finish\flask'
UPLOAD_FOLDER = Path + r'\uploads'
DATA_FOLDER =Path+ r'\dataset'
ALLOWED_EXTENSIONS = set(['txt', 'png', 'JPEG', 'JPEG','JPEG', 'gif'])
app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', template_folder='/')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    #如果無資料夾先建立資料夾
    if not(os.path.exists(UPLOAD_FOLDER)):
        os.mkdir(UPLOAD_FOLDER)

    if request.method == 'POST':
        uploaded_files = request.files.getlist("file[]")
        filenames = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],     
                                filename))
            filenames.append(filename)
    return render_template('result.html', filenames=filenames)

@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                            filename)

#分割資料
@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    JPEG_path = os.path.join(DATA_FOLDER, 'images')
    txt_path = os.path.join(DATA_FOLDER, 'labels')
    name = [os.path.join(JPEG_path, 'train'), os.path.join(JPEG_path, 'val'),
            os.path.join(txt_path, 'train'), os.path.join(txt_path, 'val')]
    
    if not(os.path.isdir(UPLOAD_FOLDER)):
        return render_template('massage.html',massage="無資料夾")
    if (os.path.isdir(DATA_FOLDER)):
        shutil.rmtree(DATA_FOLDER)
    if not(os.path.isdir(DATA_FOLDER)):
        os.mkdir(DATA_FOLDER)
        os.mkdir(JPEG_path)
        os.mkdir(txt_path)
        for i in name:
            os.mkdir(i)

    txt = glob.glob( UPLOAD_FOLDER +"/*.txt")
    train_text, val_text = train_test_split([os.path.splitext(file)[0] for file in txt], 
                                            test_size=0.2, random_state=42)
    for i in train_text:
        shutil.copy(i+".JPEG", name[0])
        shutil.copy(i+".txt", name[2])
    for i in val_text:
        shutil.copy(i+".JPEG", name[1])
        shutil.copy(i+".txt", name[3])
    
    return render_template('massage.html',massage="成功")

#訓練
@app.route('/train_html', methods=['GET', 'POST'])
def train_html():
    return render_template('train.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == "POST":
        epochs = int(request.form["epochs"])
        batch = int(request.form["batch"])
        print("epochs={}".format(epochs))
        print("batch={}".format(batch))

    submit_button_value = request.form.get("submit_button")

    if submit_button_value == "RT-DETR":
            print("rtDETR")
            if __name__ == '__main__':
            # Load a COCO-pretrained RT-DETR-l model
                model = RTDETR(r'C:\Users\USER\Desktop\LAO\Image_Recognition\finish\RTDETR\RT-DETR-main\rtdetr_pytorch\weights_detr\rtdetr-l.pt')
                model.info()
                results = model.train(data=r'C:\Users\USER\Desktop\LAO\Image_Recognition\finish\flask\config\class.yaml',imgsz=320 ,epochs=epochs, batch=batch)
                # results = model(r'C:\Users\USER\Desktop\LAO\Image_Recognition\RTDETR\RT-DETR-main\rtdetr_pytorch\1.JPEG')
            pass
    
    elif submit_button_value == "YOLOv8":
            print("yolo")
            model = YOLO(r'C:\Users\USER\Desktop\LAO\Image_Recognition\finish\RTDETR\ultralytics-main\weights\yolov8n.pt')  # load a pretrained model (recommended for training)
            # Train the model with 2 GPUs
            results = model.train(data=r'C:\Users\USER\Desktop\LAO\Image_Recognition\finish\flask\config\class.yaml', imgsz=320 ,epochs=epochs ,batch=batch ,device=0 ,workers=0)
            pass
    
    return render_template('massage.html',massage="訓練成功")

#顯示預測照
@app.route('/show_detr', methods=['GET', 'POST'])
def show_detr():
    # 指定要讀取的根目錄路徑
    root_directory = 'runs\detect'

    def list_folders_with_keyword(directory, keyword):
        folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and keyword in f]
        return folders
    
    best_folders = list_folders_with_keyword(root_directory,"train")
    best_folder=root_directory+"\\"+best_folders[-1]+r"\weights\best.pt"
    print("使用"+best_folder)
    model = RTDETR(best_folder)
    results = model.predict(source=r"C:\Users\USER\Desktop\LAO\Image_Recognition\finish\flask\dataset\images\train",imgsz=320,show=True,save=True)

    #先清空資料夾
    #目錄路徑
    if (os.path.exists('static')):
        shutil.rmtree('static')
        os.mkdir('static')
    else:
        os.mkdir('static')

    # 獲取所有資料夾列表
    # 獲取當層資料夾列表
    current_folders = list_folders_with_keyword(root_directory,"pre")

    current_folders[-1]
    #----------------
    txt = glob.glob( root_directory+'/'+current_folders[-1] +"/*.JPEG")
    for i in txt:
        shutil.copy(i, r'C:\Users\USER\Desktop\LAO\Image_Recognition\finish\flask\static')
    #----------------
    # img_path = r'C:\Users\USER\Desktop\LAO\Image_Recognition\flask\static'
    static_img=glob.glob( "./static/" +"/*.JPEG")
    return render_template('image.html',JPEG=static_img)

#顯示預測照
@app.route('/show_yolo', methods=['GET', 'POST'])
def show_yolo():
    # 指定要讀取的根目錄路徑
    root_directory = 'runs\detect'

    def list_folders_with_keyword(directory, keyword):
        folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and keyword in f]
        return folders
    
    best_folders = list_folders_with_keyword(root_directory,"train")
    best_folder=root_directory+"\\"+best_folders[-1]+r"\weights\best.pt"
    print("使用"+best_folder)
    model = YOLO(best_folder)
    results = model.predict(source=r"C:\Users\USER\Desktop\LAO\Image_Recognition\finish\flask\dataset\images\train",imgsz=320,show=True,save=True)

    #先清空資料夾
    #目錄路徑
    if (os.path.exists('static')):
        shutil.rmtree('static')
        os.mkdir('static')
    else:
        os.mkdir('static')

    # 獲取所有資料夾列表
    # 獲取當層資料夾列表
    current_folders = list_folders_with_keyword(root_directory,"pre")

    current_folders[-1]
    #----------------
    txt = glob.glob( root_directory+'/'+current_folders[-1] +"/*.JPEG")
    for i in txt:
        shutil.copy(i, r'C:\Users\USER\Desktop\LAO\Image_Recognition\finish\flask\static')
    #----------------
    # img_path = r'C:\Users\USER\Desktop\LAO\Image_Recognition\flask\static'
    static_img=glob.glob( "./static/" +"/*.JPEG")
    return render_template('image.html',JPEG=static_img)

#清空資料夾
@app.route('/clear_file', methods=['GET', 'POST'])
def clear_file():
    # if request.method == 'POST' and os.path.exists(UPLOAD_FOLDER):
    #     # 目錄路徑
    #     shutil.rmtree(UPLOAD_FOLDER)
    #     return "成功刪除資料夾"
    if request.method == 'POST' and os.path.exists(UPLOAD_FOLDER):
        # 刪除資料夾
        shutil.rmtree(UPLOAD_FOLDER)
        return render_template('massage.html',massage="成功刪除資料夾")
    elif (os.path.exists(UPLOAD_FOLDER)) == 0:
        return render_template('massage.html',massage="無此資料夾")
    else:
        return render_template('massage.html',massage="無效的請求方法")



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return render_template('/')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=8090)