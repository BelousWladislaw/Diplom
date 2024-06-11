import mysql.connector
from mysql.connector import Error
from datetime import datetime


def execute_query(host, port, database, user, password, query, params=None):
    try:
        with mysql.connector.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                connection.commit()
                print("Запрос успешно выполнен.")
    except Error as e:
        print(f"Ошибка при работе с MySQL: {e}")


def create_tables(host, port, database, user, password):
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
    execute_query(host, port, database, user, password, create_cameras_table)
    execute_query(host, port, database, user, password, create_detections_table)


def insert_camera(host, port, database, user, password, camera_name, video_path):
    insert_camera_query = '''
    INSERT INTO cameras (camera_name, video_path)
    VALUES (%s, %s)
    '''
    execute_query(host, port, database, user, password, insert_camera_query, (camera_name, video_path))


def insert_detection(host, port, database, user, password, camera_id, detection_time, photo_path):
    insert_detection_query = '''
    INSERT INTO detections (camera_id, detection_time, photo_path)
    VALUES (%s, %s, %s)
    '''
    execute_query(host, port, database, user, password, insert_detection_query, (camera_id, detection_time, photo_path))


if __name__ == '__main__':
    # Укажите правильные данные для подключения к вашей базе данных MySQL
    host = '127.0.0.1'
    port = 3306
    database = 'fire_detection'
    user = 'root'
    password = '12345'

    # Создаем таблицы
    create_tables(host, port, database, user, password)

    # Вставляем данные в таблицу cameras
    insert_camera(host, port, database, user, password, 'Camera 1', '/path/to/video1.mp4')
    insert_camera(host, port, database, user, password, 'Camera 2', '/path/to/video2.mp4')
    insert_camera(host, port, database, user, password, 'Camera 3', '/path/to/video3.mp4')

    # Вставляем данные в таблицу detections
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    insert_detection(host, port, database, user, password, 1, now, '/path/to/photo1.jpg')
    insert_detection(host, port, database, user, password, 2, now, '/path/to/photo2.jpg')
    insert_detection(host, port, database, user, password, 3, now, '/path/to/photo3.jpg')
