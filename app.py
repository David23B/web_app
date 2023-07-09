# import numpy as np
from flask import Flask, request, render_template, make_response, send_from_directory,url_for,redirect,abort,flash,send_file
# import pickle
# from feature_engineering import load_dataset,fea_importance,key_fea,train
import os
import werkzeug
from flask_dropzone import Dropzone
from flask_uploads import UploadSet,configure_uploads,patch_request_class

 
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
dropzone = Dropzone(app)
app.secret_key = "qwe"
patch_request_class(app, 1024*1024*1024)
app.config.update(
    DROPZONE_MAX_FILE_SIZE = 1024,
    DROPZONE_MAX_FILES=30,
    DROPZONE_DEFAULT_MESSAGE = "Drop files here to upload",
    
)


@app.route("/")
def home_html():
    return render_template("home.html")


@app.route("/train")
def train_html():
    return render_template("train.html",)


@app.route("/test")
def test_html():
    return render_template("test.html")


@app.route("/upload", methods=['POST'])
def upload_trainset():
    file = request.files.get('file')
    if file.filename.rsplit('.', 1)[1] == 'pkl':
        file.save('dataset/'+file.filename)
    else:
        if file.filename.rsplit('.', 1)[1] != 'csv':
            return 'CSV only!', 400
        file.save('dataset/'+file.filename)
    return "Succeed!", 202


def return_img_stream(img_local_path):
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


@app.route("/training", methods=['POST'])
def train():
    img_stream = return_img_stream('D:/GitHub/web_app/output/output.png')
    score = 0.8
    return render_template('train.html',
                           img_stream=img_stream,
                           val_score=f"the score is {score}")

    
@app.route('/train/download', methods=['GET'])
def download_file():
    # 处理文件下载的逻辑
    # 获取要下载的文件路径
    file_path = 'model/model.pkl'  # 替换为实际的文件路径
    # 发送文件给客户端
    return send_file(file_path, as_attachment=True)


@app.route('/test/download', methods=['GET'])
def download_json():
    json_path = 'output/result.json'
    return send_file(json_path, as_attachment=True)


@app.route('/testing', methods=['POST'])
def test():
    result = "a\naa\naaa\naaaa\naaaaa\naaaaaa\naaaaaaa\naaaaaaaa\naaaaaaaaa"
    return render_template('test.html', result=result)




# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory('uploads/', filename)

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/train', methods=['GET'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return '<script> alert("No selected file");window.open("/train");</script>'
#         if file and allowed_file(file.filename):
#             filename = werkzeug.utils.secure_filename(file.filename)
#             file.save('uploads/' + filename)
#             with open('test.txt','a') as f:
#                 f.write(f'{filename}')
#             return render_template("train.html")
#     return ''



























# @app.errorhandler(413)
# def too_large(e):
#     return "File is too large", 413



# # 下载
# @app.route('/train', methods=['GET'])
# def download():
#     return make_response(send_from_directory('D:/GitHub/web_app/model/', 'model.pkl', as_attachment=True))
        
        
# # 上传
# @app.route("/work", methods=['POST'])
# def upload():
#     redirect(url_for('upload'))
#     uploaded_file = request.files['file']
#     filename = secure_filename(uploaded_file.filename)
#     if filename != '':
#         file_ext = os.path.splitext(filename)[1]
#         if file_ext not in app.config['UPLOAD_EXTENSIONS']:
#             return ".csv only", 400
#         uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
#     return '', 204

# # 上传
# @app.route("/train", methods=["POST"])
# def upload():
#     uploaded_file = request.files['file']
#     filename = secure_filename(uploaded_file.filename)
#     if filename != '':
#         file_ext = os.path.splitext(filename)[1]
#         if file_ext not in app.config['UPLOAD_EXTENSIONS']:
#             return ".csv only", 400
#         uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
#     return '', 204
 
 
 
 
# @app.route('/datast/<filename>')
# def download(filename):
#     return send_from_directory(app.config['UPLOAD_PATH'], filename)

 
# @app.route("/predict_form", methods=["POST"])
# def predict_form():
#     model = pickle.load(open("model/model.pkl", "rb"))#加载模型
#     int_features = [x for x in map(eval, request.form['features'].split())]#存储用户输入的参数
#     final_features = np.array(int_features)#将用户输入的值转化为一个数组
#     pre = model([final_features])#输入模型进行预测
 
#     return render_template(
#         "index.html", prediction_text="Likely error type: {}".format(pre)#将预测值返回到Web界面，使我们看到
#     )


# @app.route("/predict_file", methods=["POST"])
# def predict_file():
#     model = pickle.load(open("model/model.pkl", "rb"))#加载模型
#     file_obj = request.files.get("test_csv")
#     if file_obj is None:
#         return "未上传文件"
#     file_obj.save('./dataset/test.csv')  # 和前端上传的文件类型要相同
#     load_dataset(dataset="test")  # 填补缺失值
    
    
# @app.route("/train", methods=["POST"])
# def upload():
#     file_obj = request.files.get("csv")
#     if file_obj is None:
#         return "未上传文件"
#     file_obj.save('./dataset/train.csv')  # 和前端上传的文件类型要相同
#     load_dataset('./dataset/train.csv')
#     all_importances = fea_importance('./dataset/train.csv')
#     selected_features = key_fea(all_importances)
#     train('./dataset/train.csv', selected_features)
#     return "训练完成"
    
 
if __name__ == "__main__":
    
    app.run(debug=True)#调试模式下运行文件，实时反应结果。仅限测试使用，生产模式下不要使用
 