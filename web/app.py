import mysql.connector
from mysql.connector import Error
from flask import Flask, g, render_template, request
from datetime import datetime

app = Flask(__name__)
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_DATABASE'] = 'fire_detection'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '12345'

def get_db():
    if 'db' not in g:
        g.db = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            port=app.config['MYSQL_PORT'],
            database=app.config['MYSQL_DATABASE'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD']
        )
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def execute_query(query, params=None):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute(query, params)
        db.commit()
        cursor.close()
    except Error as e:
        print(f"Ошибка при работе с MySQL: {e}")

def create_tables():
    create_cameras_table = '''
    CREATE TABLE IF NOT EXISTS cameras (
        id INT AUTO_INCREMENT PRIMARY KEY,
        camera_name VARCHAR(255) NOT NULL,
        video_path VARCHAR(255) NOT NULL
    )
    '''
    create_detections_table = '''
    CREATE TABLE IF NOT EXISTS detections (
        id INT AUTO_INCREMENT PRIMARY KEY,
        camera_id INT NOT NULL,
        detection_time DATETIME NOT NULL,
        photo_path VARCHAR(255) NOT NULL,
        FOREIGN KEY (camera_id) REFERENCES cameras(id)
    )
    '''
    execute_query(create_cameras_table)
    execute_query(create_detections_table)

def insert_camera(camera_name, video_path):
    insert_camera_query = '''
    INSERT INTO cameras (camera_name, video_path)
    VALUES (%s, %s)
    '''
    execute_query(insert_camera_query, (camera_name, video_path))

def insert_detection(camera_id, detection_time, photo_path):
    insert_detection_query = '''
    INSERT INTO detections (camera_id, detection_time, photo_path)
    VALUES (%s, %s, %s)
    '''
    execute_query(insert_detection_query, (camera_id, detection_time, photo_path))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/grid')
def grid():
    return render_template('grid.html')

@app.route('/reports', methods=['GET', 'POST'])
def reports():
    reports_data = []
    start_date = None
    end_date = None

    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        db = get_db()
        cursor = db.cursor()
        query = '''
        SELECT c.camera_name, d.detection_time, d.photo_path
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        WHERE d.detection_time BETWEEN %s AND %s
        '''
        cursor.execute(query, (start_date, end_date))
        reports_data = cursor.fetchall()
        cursor.close()

    return render_template('reports.html', reports_data=reports_data, start_date=start_date, end_date=end_date)

if __name__ == '__main__':
    with app.app_context():
        create_tables()
        print("Таблицы успешно созданы")
        # Вставляем данные в таблицу cameras
        insert_camera('Camera 1', '/path/to/video1.mp4')
        insert_camera('Camera 2', '/path/to/video2.mp4')
        insert_camera('Camera 3', '/path/to/video3.mp4')

        # Вставляем данные в таблицу detections
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        insert_detection(1, now, 'photo1.jpg')
        insert_detection(2, now, 'photo2.jpg')
        insert_detection(3, now, 'photo3.jpg')
        print("Данные успешно вставлены")
    app.run(debug=True, port=5000)




