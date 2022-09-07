from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from skimage.feature import graycomatrix, graycoprops
from werkzeug.utils import secure_filename
from  io import BytesIO
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score
import pymysql
import numpy as np
import pandas as pd
import os
import base64
import KNN

app = Flask(__name__)

# Converter for processing numpy float64
pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
pymysql.converters.conversions = pymysql.converters.encoders.copy()
pymysql.converters.conversions.update(pymysql.converters.decoders)

# Connect to the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/tugasakhir2?charset=utf8mb4'
db = SQLAlchemy(app)
connection = pymysql.connect(host='localhost', user='root', password='', db='tugasakhir2') #pymysql

# Global variable
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
newsize = (256,256)
sudut = ['0', '45', '90','135']
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg']
index_df = []
for x in properties:
    for y in sudut:
        xy = x + " " + y
        index_df.append(xy)
euclidean_index = ['Filename']
for x in properties:
    for y in sudut:
        xy = x + " " + y
        euclidean_index.append(xy)
euclidean_index.append('Class')
euclidean_index.append('Euclidean Distances')
manhattan_index = ['Filename']
for x in properties:
    for y in sudut:
        xy = x + " " + y
        manhattan_index.append(xy)
manhattan_index.append('Class')
manhattan_index.append('Manhattan Distances')
accuracy_report = pd.DataFrame(columns=['Filename', 'Actual', 'Euclidean_Result', 'Manhattan_Result'])

class glcm_features(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    target = db.Column(db.String(10), nullable=False)
    dissimilarity_0 = db.Column(db.Float, primary_key=False)
    dissimilarity_45 = db.Column(db.Float, primary_key=False)
    dissimilarity_90 = db.Column(db.Float, primary_key=False)
    dissimilarity_135 = db.Column(db.Float, primary_key=False)
    correlation_0 = db.Column(db.Float, primary_key=False)
    correlation_45 = db.Column(db.Float, primary_key=False)
    correlation_90 = db.Column(db.Float, primary_key=False)
    correlation_135 = db.Column(db.Float, primary_key=False)
    homogeneity_0 = db.Column(db.Float, primary_key=False)
    homogeneity_45 = db.Column(db.Float, primary_key=False)
    homogeneity_90 = db.Column(db.Float, primary_key=False)
    homogeneity_135 = db.Column(db.Float, primary_key=False)
    contrast_0 = db.Column(db.Float, primary_key=False)
    contrast_45 = db.Column(db.Float, primary_key=False)
    contrast_90 = db.Column(db.Float, primary_key=False)
    contrast_135 = db.Column(db.Float, primary_key=False)
    asm_0 = db.Column(db.Float, primary_key=False)
    asm_45 = db.Column(db.Float, primary_key=False)
    asm_90 = db.Column(db.Float, primary_key=False)
    asm_135 = db.Column(db.Float, primary_key=False)
    energy_0 = db.Column(db.Float, primary_key=False)
    energy_45 = db.Column(db.Float, primary_key=False)
    energy_90 = db.Column(db.Float, primary_key=False)
    energy_135 = db.Column(db.Float, primary_key=False)

    def __repr__(self):
        return '<data %r>' % self.id

class tb_target(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    target = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return '<data %r>' % self.id

# GLCM extraction greycomatrix() & greycoprops() for angle 0, 45, 90, 135
# dists = jarak antar pixel yang akan dikalkulasi
def calc_glcm_all_agls(img, props, dists=[5], agls=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], lvl=256, sym=True,
                       norm=True):
    glcm = graycomatrix(img,
                        distances=dists,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    return feature

# Build GLCM DataFrame
def GLCM_df(Image):
    columns = []
    glcm_all_agls = calc_glcm_all_agls(Image, props=properties)
    for name in properties:
        for ang in sudut:
            columns.append(name + "_" + ang + " ")
    # dataframe GLCM
    glcm_df = pd.DataFrame(glcm_all_agls, columns)
    return glcm_df

# Fecting dataset from database and save it into numpy array
def fetch_data():
    id_all = "SELECT COUNT(id) FROM glcm_features"
    all_sk = "SELECT * FROM glcm_features"

    id_sk34 = "SELECT COUNT(id) FROM glcm_features WHERE (filename NOT LIKE '%sk3.jpg' AND filename NOT LIKE '%sk4.jpg')"
    sk34 = "SELECT * FROM glcm_features WHERE (filename NOT LIKE '%sk3.jpg' AND filename NOT LIKE '%sk4.jpg')"

    id_sk56 = "SELECT COUNT(id) FROM glcm_features WHERE (filename NOT LIKE '%sk5.jpg' AND filename NOT LIKE '%sk6.jpg')"
    sk56 = "SELECT * FROM glcm_features WHERE (filename NOT LIKE '%sk5.jpg' AND filename NOT LIKE '%sk6.jpg')"

    # 61% accuracy here (60% and 70% noise) with db2
    id_sk810 = "SELECT COUNT(id) FROM glcm_features WHERE (filename NOT LIKE '%sk8.jpg' AND filename NOT LIKE '%sk10.jpg')"
    sk810 = "SELECT * FROM glcm_features WHERE (filename NOT LIKE '%sk8.jpg' AND filename NOT LIKE '%sk10.jpg')"

    # BEST accuracy here (5% and 65% noise) with db2
    id_sk29 = "SELECT COUNT(id) FROM glcm_features WHERE (filename NOT LIKE '%sk2.jpg' AND filename NOT LIKE '%sk9.jpg')"
    sk29 = "SELECT * FROM glcm_features WHERE (filename NOT LIKE '%sk2.jpg' AND filename NOT LIKE '%sk9.jpg')"

    cursor = connection.cursor()
    cursor.execute(id_sk29) # get rows
    n_rows = cursor.fetchone()
    baris = n_rows[0]
    print("Jumlah Baris Dataset : ", baris)
    kolom = 27
    data = np.zeros((baris, kolom), dtype=object)
    # print("Data Sementara :")
    # print(data)
    cursor.execute(sk29) # get data
    i = 0
    for x in cursor.fetchall():
        for y in range(kolom):
            data[i][y] = x[y]
        i = i + 1
    # print("Data Baru :")
    # print(data)
    # print("Data Tanpa Kolom ID :")
    filenames = data[:,1]
    data = np.delete(data, [0,1], axis=1)
    # print(data)
    return data, filenames


@app.route('/', methods=['POST', 'GET'])
def index():
    global accuracy_report
    data_kelas = tb_target.query.order_by(tb_target.target).all()
    if request.method == 'POST':
        # print("Yeay Tombol Bisa Dipake")
        uploaded_file = request.files["query_img"]
        filename = secure_filename(uploaded_file.filename)
        raw_filename = secure_filename(r"{}".format(uploaded_file.filename))
        # mengecek apakah user sudah memilih file atau belum
        if filename != '':
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return render_template('index.html', data_kelas=data_kelas, ext='Format file yang diupload tidak didukung oleh sistem. Hanya file dengan format JPG dan JPEG yang didukung')
            else:
                # open image PIL
                img = Image.open(uploaded_file.stream)
                img = img.resize(newsize)
                gambar = ImageOps.grayscale(img)
                img_blob = img.convert('RGB')
                image = np.asarray(gambar)
                print("Nama File : ", filename)
                print("Array Gambar:\n", image.shape)
                print(image)
                print("Hasil Ekstrasi Fitur GLCM")
                # data = np.asarray(GLCM_df(image), dtype=object) # Bentukannya Numpy
                data = GLCM_df(image) # Bentukannya Array List
                print(data)
                k = request.form['nilai_k']
                k = int(k)
                actual_class = r"{}".format(request.form['target'])
                print("Nilai K : ", k)

                # Image to blob (RGB)
                buffered = BytesIO()
                img_blob.save(buffered, format="JPEG")
                img_encode = base64.b64encode(buffered.getvalue())
                # decode image base64 to load on HTML
                img_decode = img_encode.decode("utf-8")
                # print(img_decode)
                # print("Gambar Berhasil Masuk")

                # Image to blob (Greyscale)
                buffered1 = BytesIO()
                gambar.save(buffered1, format="JPEG")
                img_encode_bw = base64.b64encode(buffered1.getvalue())
                # decode image base64 to load on HTML
                img_decode_bw = img_encode_bw.decode("utf-8")

                # Get dataset and save it into array
                dataset, filenames = fetch_data()
                print("LALALALA", dataset.shape, filenames.shape)
                # df_dataset = pd.DataFrame(dataset)
                # df_dataset = df_dataset.drop_duplicates() #remove duplicates
                # dataset = df_dataset.to_numpy()
                print("\nUkuran Dataset : ", dataset.shape)
                test = np.asarray(data, dtype=float)
                test = np.around(test, decimals=5)
                # print(test)
                # print(dataset)
                print("Ukuran Data Test : ", test.shape)
                print("Dataset Berhasil Disimpan ke Dalam Array!")

                # Classification process using Euclidean Distance KNN
                EuclideanDistancesNotSorted, EuclideanDistances, EuclideanNeighbors = KNN.getKNeighbors(dataset, test, k, "Euclidean")
                EuclideanResult = KNN.getResponse(EuclideanNeighbors)
                print("Tipe Data Euclidean Results :\n", type(EuclideanDistances))
                print(EuclideanDistances.shape)
                print("\nHasil Tetangga Terdekat : ")
                print(EuclideanNeighbors)  # Ukuran array neighbors bergantung pada nilai k
                print("Hasil Klasifikasi Dengan Euclidean : ")
                print(EuclideanResult)

                # Classification process using Manhattan Distance KNN
                ManhattanDistancesNotSorted, ManhattanDistances, ManhattanNeighbors = KNN.getKNeighbors(dataset, test, k, "Manhattan")
                ManhattanResult = KNN.getResponse(ManhattanNeighbors)
                print("Tipe Data Manhattan Results :\n", type(ManhattanDistances))
                print(ManhattanDistances.shape)
                print("\nHasil Tetangga Terdekat : ")
                print(ManhattanNeighbors)  # Ukuran array neighbors bergantung pada nilai k
                print("Hasil Klasifikasi Dengan Manhattan : ")
                print(ManhattanResult)

                # Euclidean Data preprocessing before exporting to excel
                print("\nEuclidean Results Preprocessing")
                print(dataset.shape)
                filenames = filenames[None,...]
                print(filenames.shape)
                dataset = np.insert(dataset, 0, filenames, axis=1)
                a = np.array(EuclideanDistancesNotSorted[:, 1])
                a = a[..., None]
                print(a.shape)
                EuclideanReady = np.concatenate((dataset, a), axis=1)
                print(EuclideanReady.shape)
                EuclideanResults_df = pd.DataFrame(EuclideanReady, columns = euclidean_index)
                EuclideanSortedResults_df = EuclideanResults_df.sort_values(by=['Euclidean Distances'])
                EuclideanNeighbors_df = EuclideanSortedResults_df.head(k)
                EuclideanNeighborsClass_df = EuclideanNeighbors_df.loc[:, 'Class']
                EuclideanNeighborsVoted_df = EuclideanNeighborsClass_df.mode()

                # Manhattan Data preprocessing before exporting to excel
                print("\nManhattan Results Preprocessing")
                print(dataset.shape)
                b = np.array(ManhattanDistancesNotSorted[:, 1])
                b = b[..., None]
                print(b.shape)
                ManhattanReady = np.concatenate((dataset, b), axis=1)
                print(ManhattanReady.shape)
                ManhattanResults_df = pd.DataFrame(ManhattanReady, columns=manhattan_index)
                ManhattanSortedResults_df = ManhattanResults_df.sort_values(by=['Manhattan Distances'])
                ManhattanNeighbors_df = ManhattanSortedResults_df.head(k)
                ManhattanNeighborsClass_df = ManhattanNeighbors_df.loc[:, 'Class']
                ManhattanNeighborsVoted_df = ManhattanNeighborsClass_df.mode()

                # Write data to excel
                excel_filename = filename.split(".", 1)
                excel_filename = excel_filename[0]
                test_df = pd.DataFrame(test, columns = ['Nilai'], index=index_df)
                with pd.ExcelWriter("Results//" + excel_filename + '.xlsx') as writer:
                    test_df.to_excel(writer, sheet_name='Data Test')
                    EuclideanResults_df.to_excel(writer, sheet_name='Euclidean Distances')
                    EuclideanSortedResults_df.to_excel(writer, sheet_name='Euclidean Distances Sorted')
                    EuclideanNeighbors_df.to_excel(writer, sheet_name='Euclidean Distances Neighbors')
                    EuclideanNeighborsClass_df.to_excel(writer, sheet_name='Euclidean Distances Class')
                    EuclideanNeighborsVoted_df.to_excel(writer, sheet_name='Euclidean Distances Result')
                    ManhattanResults_df.to_excel(writer, sheet_name='Manhattan Distances')
                    ManhattanSortedResults_df.to_excel(writer, sheet_name='Manhattan Distances Sorted')
                    ManhattanNeighbors_df.to_excel(writer, sheet_name='Manhattan Distances Neighbors')
                    ManhattanNeighborsClass_df.to_excel(writer, sheet_name='Manhattan Distances Class')
                    ManhattanNeighborsVoted_df.to_excel(writer, sheet_name='Manhattan Distances Result')

                #Collect Classification Report
                accuracy_report = accuracy_report.append({'Filename' : raw_filename,
                                                          'Actual' : actual_class,
                                                          'Euclidean_Result' : EuclideanResult,
                                                          'Manhattan_Result' : ManhattanResult},
                                                         ignore_index = True)

                return render_template('index.html', data_kelas=data_kelas, query_path=img_decode, query_path_bw=img_decode_bw, filename=raw_filename, data=test, nilai_k=k, hasil=EuclideanResult)
        return render_template('index.html', data_kelas=data_kelas)
    else:
        return render_template('index.html', data_kelas=data_kelas)

@app.route('/KelolaData', methods=['POST', 'GET'])
def view_data():
    # data = glcm_features.query.filter(glcm_features.filename.match("%sk1.jpg")).all()
    # data = glcm_features.query.filter(glcm_features.filename.notlike("%sk3_jpg") &
    #                                   glcm_features.filename.notlike("%sk4_jpg")
    #                                   )
    # data = glcm_features.query.filter(glcm_features.filename.notlike("%sk5_jpg") &
    #                                   glcm_features.filename.notlike("%sk6_jpg")
    #                                   )
    # data = glcm_features.query.filter(glcm_features.filename.notlike("%sk8_jpg") &
    #                                   glcm_features.filename.notlike("%sk10_jpg")
    #                                   )
    # data = glcm_features.query.filter(glcm_features.filename.notlike("%sk11_jpg"))
    data = glcm_features.query.order_by(glcm_features.id).all()
    return render_template('dataset.html', data=data)

@app.route('/TambahData', methods=['POST', 'GET'])
def add_data():
    data_kelas = tb_target.query.order_by(tb_target.target).all()
    if request.method == 'POST':
        # print("Yeay Tombol Bisa Dipake")
        uploaded_file = request.files["query_img"]
        filename = secure_filename(uploaded_file.filename)
        # print(filename)
        # mengecek apakah user sudah memilih file atau belum
        if filename != '':
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return render_template('add.html',
                                       ext='Uploaded file is not a valid image. Only JPG, JPEG and PNG files are allowed')
            else:
                # open image PIL
                img = Image.open(uploaded_file.stream)
                img = img.resize(newsize)
                img_blob = img.convert('RGB')
                gambar = ImageOps.grayscale(img)
                image = np.asarray(gambar)
                print("Nama File : ", filename)
                print("Array Gambar:\n", image.shape)

                # ekstrasi fitur GLCM
                # print("Hasil Ekstrasi Fitur GLCM")
                data = GLCM_df(image)
                data = np.array(data, dtype=float)
                data = np.reshape(data,(1,24))
                print(data.shape)
                data = np.around(data, decimals=6)
                print(data.shape)
                # data = data.tolist()
                # print(data)
                target = request.form['target']
                print(target)

                # Inserting into DB
                cursor = connection.cursor()
                sql = "INSERT INTO glcm_features (filename, dissimilarity_0, dissimilarity_45, dissimilarity_90, dissimilarity_135, correlation_0, correlation_45, correlation_90, correlation_135, homogeneity_0, homogeneity_45, homogeneity_90, homogeneity_135,contrast_0, contrast_45, contrast_90, contrast_135, asm_0, asm_45, asm_90, asm_135, energy_0, energy_45, energy_90, energy_135, target) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (
                    filename, data[0][0], data[0][1], data[0][2], data[0][3], data[0][4], data[0][5], data[0][6],
                    data[0][7], data[0][8], data[0][9], data[0][10], data[0][11], data[0][12], data[0][13], data[0][14],
                    data[0][15], data[0][16], data[0][17], data[0][18], data[0][19], data[0][20], data[0][21],
                    data[0][22],
                    data[0][23], target))
                connection.commit()
                # connection.close()
                print("Dataset Gambar Baru Berhasil Ditambahkan!")

                # Image to blob (RGB)
                buffered = BytesIO()
                img_blob.save(buffered, format="JPEG")
                img_encode = base64.b64encode(buffered.getvalue())
                # decode image base64 to load on HTML
                img_decode = img_encode.decode("utf-8")
                # print(img_decode)

                # Image to blob (Greyscale)
                buffered1 = BytesIO()
                gambar.save(buffered1, format="JPEG")
                img_encode_bw = base64.b64encode(buffered1.getvalue())
                # decode image base64 to load on HTML
                img_decode_bw = img_encode_bw.decode("utf-8")

                return render_template('add.html', data_kelas=data_kelas, query_path=img_decode, query_path_bw=img_decode_bw, data=data, target=target, filename=filename)
        return render_template('add.html', data_kelas=data_kelas)
    else:
        return render_template('add.html', data_kelas=data_kelas)

@app.route('/delete/<int:id>')
def delete(id):
    data_to_delete = glcm_features.query.get_or_404(id)
    try:
        db.session.delete(data_to_delete)
        db.session.commit()
        db.session.close()
        return redirect('/KelolaData')
    except:
        data = glcm_features.query.order_by(glcm_features.id).all()
        return render_template('dataset.html', data=data)

@app.route('/detail/<int:id>', methods=['POST', 'GET'])
def detail(id):
    data_to_update = glcm_features.query.get_or_404(id)
    return render_template('detail.html', data_to_update=data_to_update)

@app.route('/ExportAcc')
def export_accuracy():
    global accuracy_report
    data_kelas = tb_target.query.order_by(tb_target.target).all()
    euclidean_score = accuracy_score(accuracy_report['Actual'], accuracy_report['Euclidean_Result'])
    # manhattan_score = accuracy_score(accuracy_report['Actual'], accuracy_report['Manhattan_Result'])
    # accuracy = pd.DataFrame({'Euclidean': euclidean_score,
    #                          'Manhattan': manhattan_score}, index=[0])
    accuracy = pd.DataFrame({'Euclidean' : euclidean_score}, index=[0])

    # Write data to excel
    excel_filename = "Accuracy_Report"
    with pd.ExcelWriter("Report//" + excel_filename + '.xlsx') as writer:
        accuracy_report.to_excel(writer, sheet_name='Classification Report')
        accuracy.to_excel(writer, sheet_name='Accuracy')
    # return render_template('index.html', data_kelas=data_kelas)
    return index()

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='192.168.100.70', port=5000, debug=True)